"""Experiments 5 and 6: six-lead, native 500 Hz CLBBB/ILBBB classification."""
from pathlib import Path
import argparse
import ast
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import wfdb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from experiment_utils import sha256, set_seed
from ecg_resnet import ECGResNet

LEADS = ['II', 'III', 'aVR', 'aVF', 'V3', 'V4']
LABELS = ['CLBBB', 'ILBBB']
SPLITS = ['train', 'val', 'test']
SOURCE = Path(__file__).parent / 'ecg_resnet.py'
DATASET_SOURCES = {
    'dataset1': dict(branch='feature/dataset1-12lead-clean',
                     commit='4d82a777d7b811ea35e57d1abfc9cfb5c8128f6e',
                     builder='learning_pipeline/02_build_dataset1.py'),
    'dataset2': dict(branch='kevin/experiment2-dataset2-all12',
                     commit='3e01b5ebedc1ec2db3f6ac71714d1d86b0f28127',
                     builder='experiment2/01_build_dataset2.py'),
}


def dataset1_metadata(root):
    """Select code-presence LBBB positives and diagnostic-only Normal negatives."""
    frame = pd.read_csv(root / 'ptbxl_database.csv')
    statements = pd.read_csv(root / 'scp_statements.csv', index_col=0)
    diagnostic = set(statements.index[statements.diagnostic == 1]) - {'NORM'}
    codes = frame.scp_codes.map(ast.literal_eval)
    for label in LABELS:
        frame[label] = codes.map(lambda c: int(label in c))
    normal = codes.map(lambda c: 'NORM' in c and not (set(c) & diagnostic))
    return frame[normal | frame[LABELS].any(axis=1)].copy()


def dataset2_metadata(root):
    """Selection and assertions from the pinned Dataset 2 builder; HR path only."""
    frame = pd.read_csv(root / 'ptbxl_database.csv')
    if not frame.ecg_id.is_unique:
        raise ValueError('Duplicate ECG IDs')
    codes = frame.scp_codes.apply(ast.literal_eval)
    frame['CLBBB'] = codes.map(lambda c: int('CLBBB' in c))
    frame['ILBBB'] = codes.map(lambda c: int('ILBBB' in c))
    columns = ['ecg_id', 'patient_id', 'CLBBB', 'ILBBB', 'strat_fold', 'filename_hr']
    dataset = frame[columns].copy()
    if not dataset.strat_fold.between(1, 10).all():
        raise ValueError('Invalid official folds')
    if not dataset.groupby('patient_id').strat_fold.nunique().eq(1).all():
        raise ValueError('Patient split leakage')
    if ((dataset.CLBBB == 1) & (dataset.ILBBB == 1)).any():
        raise ValueError('Dataset 2 source builder rejects overlapping labels')
    return dataset


class ECGArrayDataset(Dataset):
    """Read disk-backed waveforms per item, matching the Dataset 2 source loader."""
    def __init__(self, signals, labels):
        self.x = signals
        self.y = torch.from_numpy(np.array(labels, dtype=np.float32, copy=True))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return torch.from_numpy(np.array(self.x[index], copy=True)), self.y[index]


def validate_metadata(frame):
    if frame.empty or frame[['ecg_id', 'patient_id', 'strat_fold', 'filename_hr']].isna().any().any():
        raise ValueError('Empty cohort or missing identifiers')
    if frame.ecg_id.duplicated().any() or frame.filename_hr.duplicated().any():
        raise ValueError('Duplicate recording')
    for filename in frame.filename_hr:
        relative = Path(filename)
        if relative.is_absolute() or '..' in relative.parts or not relative.parts or relative.parts[0] != 'records500':
            raise ValueError('Expected records500 waveform path')
    if not frame.strat_fold.isin(range(1, 11)).all():
        raise ValueError('Invalid official fold')
    if frame.groupby('patient_id').strat_fold.nunique().max() != 1:
        raise ValueError('Patient leakage between folds')
    if not np.isin(frame[LABELS].to_numpy(), [0, 1]).all():
        raise ValueError('Expected two binary labels')


def prepare(root, out, dataset='dataset1'):
    frame = dataset1_metadata(root) if dataset == 'dataset1' else dataset2_metadata(root)
    validate_metadata(frame)
    out.mkdir(parents=True, exist_ok=False)
    for split, folds in zip(SPLITS, [range(1, 9), [9], [10]]):
        rows = frame[frame.strat_fold.isin(folds)]
        signals = np.lib.format.open_memmap(out / f'X_{split}.npy', mode='w+',
                                           dtype=np.float32, shape=(len(rows), 6, 5000))
        for index, row in enumerate(rows.itertuples()):
            relative = Path(row.filename_hr)
            if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'records500':
                raise ValueError('Invalid raw waveform path')
            signal, fields = wfdb.rdsamp(str(root / relative))
            names = [n.upper() for n in fields['sig_name']]
            if fields['fs'] != 500 or signal.shape != (5000, 12) or len(set(names)) != 12:
                raise ValueError('Unexpected waveform contract')
            indices = [names.index(n.upper()) for n in LEADS]
            if any(fields['units'][i].lower() != 'mv' for i in indices) or not np.isfinite(signal).all():
                raise ValueError('Invalid amplitudes or units')
            signals[index] = signal[:, indices].T.astype(np.float32)
            if (index + 1) % 1000 == 0:
                print(f'{dataset} {split}: {index + 1}/{len(rows)}', flush=True)
        signals.flush()
        del signals
        np.save(out / f'y_{split}.npy', rows[LABELS].to_numpy(dtype=np.float32))
        rows.to_csv(out / f'{split}.csv', index=False)
    policy = ('dataset1-v1: code presence; diagnostic-only Normal filter; comorbid LBBB permitted'
              if dataset == 'dataset1' else
              'dataset2-v1: all source ECGs; code-presence labels; source rejects overlapping labels')
    (out / 'contract.json').write_text(json.dumps(dict(dataset=dataset, leads=LEADS,
        labels=LABELS, fs=500, samples=5000, normalization='none',
        source_column='filename_hr', waveform_directory='records500', units='mV',
        policy=policy, source=DATASET_SOURCES[dataset],
        metadata_sha256=sha256(root / 'ptbxl_database.csv'),
        statements_sha256=sha256(root / 'scp_statements.csv')), indent=2))


def load(directory, expected):
    contract = json.loads((directory / 'contract.json').read_text())
    for k, v in dict(dataset=expected, leads=LEADS, labels=LABELS, fs=500, samples=5000, normalization='none',
                     source_column='filename_hr', waveform_directory='records500', units='mV').items():
        if contract.get(k) != v:
            raise ValueError(f'Wrong export contract: {k}')
    if not contract.get('policy'):
        raise ValueError('Labeling policy provenance required')
    frames = [pd.read_csv(directory / f'{s}.csv') for s in SPLITS]
    validate_metadata(pd.concat(frames))
    data, counts, fingerprints = {}, {}, {}
    for split, frame in zip(SPLITS, frames):
        expected_fold = frame.strat_fold.between(1, 8) if split == 'train' else frame.strat_fold.eq(9 if split == 'val' else 10)
        if frame.empty or not expected_fold.all():
            raise ValueError('Incorrect held-out split')
        x = np.load(directory / f'X_{split}.npy', mmap_mode='r', allow_pickle=False)
        y = np.load(directory / f'y_{split}.npy', allow_pickle=False)
        if (x.dtype != np.float32 or x.shape != (len(frame), 6, 5000)
                or not all(np.isfinite(x[start:start+256]).all()
                           for start in range(0, len(x), 256))):
            raise ValueError('Invalid six-lead signals')
        if y.shape != (len(frame), 2) or not np.array_equal(y, frame[LABELS].to_numpy()):
            raise ValueError('Label alignment failure')
        data[split] = ECGArrayDataset(x, y)
        counts[split] = dict(recordings=len(frame), patients=frame.patient_id.nunique(),
            **{label: dict(positive=int(y[:, i].sum()), negative=int(len(y)-y[:, i].sum())) for i, label in enumerate(LABELS)})
        for filename in [f'X_{split}.npy', f'y_{split}.npy', f'{split}.csv']:
            fingerprints[filename] = sha256(directory / filename)
    fingerprints['contract.json'] = sha256(directory / 'contract.json')
    return data, dict(contract=contract, counts=counts, sha256=fingerprints, patient_overlap=0), frames


def verify_cross_dataset(frames1, frames2):
    """Dataset 1 must be the same-record subset of Dataset 2 within each fold."""
    checks = {}
    columns = ['patient_id', 'strat_fold', 'CLBBB', 'ILBBB', 'filename_hr']
    for split, first, second in zip(SPLITS, frames1, frames2):
        second = second.set_index('ecg_id')
        if not first.ecg_id.isin(second.index).all():
            raise ValueError(f'{split}: Dataset 1 is not a subset of Dataset 2')
        for column in columns:
            if not np.array_equal(first[column], second.loc[first.ecg_id, column]):
                raise ValueError(f'{split}: cross-dataset mismatch in {column}')
        checks[split] = dict(dataset1_subset=True, matching_records=len(first))
    return checks


def metrics(y, scores):
    result = {}
    for i, label in enumerate(LABELS):
        true, pred = y[:, i], scores[:, i] >= .5
        result[label] = dict(accuracy=accuracy_score(true, pred),
            precision=precision_score(true, pred, zero_division=0),
            recall=recall_score(true, pred, zero_division=0), f1=f1_score(true, pred, zero_division=0),
            confusion_matrix=confusion_matrix(true, pred, labels=[0, 1]).tolist())
    return result


def run(args):
    loaded = {'dataset1': load(args.dataset1, 'dataset1')}
    if args.dataset2:
        loaded['dataset2'] = load(args.dataset2, 'dataset2')
    if args.train not in loaded:
        raise ValueError('Dataset 2 export unavailable; no substitution')
    subset_checks = (verify_cross_dataset(loaded['dataset1'][2], loaded['dataset2'][2])
                     if 'dataset2' in loaded else {})
    # Cross-dataset evaluation must not introduce patients seen in training/validation.
    training_frames = loaded[args.train][2]
    seen = set(pd.concat(training_frames[:2]).patient_id)
    leakage_checks = {}
    for name, (_, _, frames) in loaded.items():
        if seen & set(frames[2].patient_id):
            raise ValueError(f'Patient leakage into {name} test')
        leakage_checks[name] = dict(test_patients=len(set(frames[2].patient_id)),
                                   overlap_with_training_or_validation=0)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'leakage_checks.json').write_text(json.dumps(
        dict(test_patient_checks=leakage_checks, subset_checks=subset_checks), indent=2))
    set_seed(42)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    model = ECGResNet().to(device)
    data = loaded[args.train][0]
    loaders = {s: DataLoader(d, batch_size=32, shuffle=s == 'train') for s, d in data.items()}
    y = data['train'].y
    positives = y.sum(0)
    if (positives == 0).any() or (positives == len(y)).any():
        raise ValueError('Each training output requires positive and negative examples')
    weights = ((len(y)-positives)/positives).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    history, best, stale = [], float('inf'), 0
    provenance = {k: v[1] for k, v in loaded.items()}
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    config = dict(train_dataset=args.train, seed=42, batch_size=32, max_epochs=args.epochs,
        patience=7, learning_rate=.001, optimizer='Adam', device=str(device),
        fs=500, samples=5000, source_column='filename_hr', waveform_directory='records500',
        pos_weight=weights.cpu().tolist(), selection='lowest validation weighted BCE', threshold=.5)
    (args.output / 'training_config.json').write_text(json.dumps(config, indent=2))
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if checkpoint['training_config']['train_dataset'] != args.train:
            raise ValueError('Checkpoint training dataset differs')
        if checkpoint['provenance'][args.train] != provenance[args.train]:
            raise ValueError('Training export differs from checkpoint provenance')
        if checkpoint['architecture_sha256'] != sha256(SOURCE):
            raise ValueError('Checkpoint architecture source differs')
        if checkpoint['leads'] != LEADS or checkpoint['labels'] != LABELS:
            raise ValueError('Checkpoint lead or label order differs')
    for epoch in range(1, 1 if args.checkpoint else args.epochs+1):
        losses = {}
        for split in ['train', 'val']:
            model.train(split == 'train')
            total = 0
            with torch.set_grad_enabled(split == 'train'):
                for x, target in loaders[split]:
                    x, target = x.to(device), target.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(x), target)
                    if split == 'train':
                        loss.backward()
                        optimizer.step()
                    total += loss.item()*len(x)
            losses[split] = total/len(data[split])
        history.append(dict(epoch=epoch, **losses))
        if losses['val'] < best:
            best, stale = losses['val'], 0
            torch.save(dict(model_state_dict=model.state_dict(), epoch=epoch, val_loss=best,
                pos_weight=weights.cpu(), seed=42, leads=LEADS, labels=LABELS, training_config=config,
                provenance=provenance, architecture_sha256=sha256(SOURCE)), args.output / 'best_model.pt')
        else:
            stale += 1
        (args.output / 'history.json').write_text(json.dumps(history, indent=2))
        print(history[-1], flush=True)
        if stale >= 7:
            break
    if not args.checkpoint:
        checkpoint = torch.load(args.output / 'best_model.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    results = dict(fs=500, samples=5000, leads=LEADS, labels=LABELS, train_dataset=args.train, best_epoch=checkpoint['epoch'], threshold=.5,
        confusion_matrix_order='true rows / predicted columns: negative, positive',
        dataset2_status='available' if args.dataset2 else 'pending: approved team Dataset 2 unavailable', tests={})
    with torch.no_grad():
        for name, (datasets, _, _) in loaded.items():
            scores = np.concatenate([torch.sigmoid(model(x.to(device))).cpu().numpy()
                for x, _ in DataLoader(datasets['test'], batch_size=32)])
            truth = datasets['test'].y.numpy()
            results['tests'][name] = metrics(truth, scores)
            np.savez(args.output / f'{name}_test_predictions.npz', truth=truth, scores=scores)
    (args.output / 'metrics.json').write_text(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--raw-root', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--dataset', choices=['dataset1', 'dataset2'], default='dataset1')
    train = sub.add_parser('train')
    train.add_argument('--dataset1', type=Path, required=True)
    train.add_argument('--dataset2', type=Path)
    train.add_argument('--train', choices=['dataset1', 'dataset2'], default='dataset1')
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--epochs', type=int, default=30)
    train.add_argument('--device', default='cpu')
    train.add_argument('--checkpoint', type=Path, help='Evaluate an existing best checkpoint without retraining')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.raw_root, args.output, args.dataset)
    else:
        if args.epochs < 1:
            parser.error('epochs must be positive')
        run(args)

if __name__ == '__main__':
    main()
