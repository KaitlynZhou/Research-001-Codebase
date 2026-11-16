"""Experiments 5 and 6: six-lead, 500 Hz CLBBB/ILBBB classification.

Dataset 2 requires an approved export; no cohort is inferred or substituted.
"""
from pathlib import Path
import argparse
import ast
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import wfdb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from experiment_utils import sha256, set_seed
from ecg_resnet import ECGResNet

LEADS = ['II', 'III', 'aVR', 'aVF', 'V3', 'V4']
LABELS = ['CLBBB', 'ILBBB']
SPLITS = ['train', 'val', 'test']
SOURCE = Path(__file__).parent / 'ecg_resnet.py'


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


def prepare(root, out):
    frame = dataset1_metadata(root)
    validate_metadata(frame)
    out.mkdir(parents=True, exist_ok=False)
    for split, folds in zip(SPLITS, [range(1, 9), [9], [10]]):
        rows = frame[frame.strat_fold.isin(folds)]
        signals = []
        for row in rows.itertuples():
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
            signals.append(signal[:, indices].T.astype(np.float32))
        np.save(out / f'X_{split}.npy', np.stack(signals))
        np.save(out / f'y_{split}.npy', rows[LABELS].to_numpy(dtype=np.float32))
        rows.to_csv(out / f'{split}.csv', index=False)
    (out / 'contract.json').write_text(json.dumps(dict(dataset='dataset1', leads=LEADS,
        labels=LABELS, fs=500, samples=5000, normalization='none',
        source_column='filename_hr', waveform_directory='records500', units='mV',
        policy='dataset1-v1: code presence; diagnostic-only Normal filter; comorbid LBBB permitted',
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
        x = np.load(directory / f'X_{split}.npy', allow_pickle=False)
        y = np.load(directory / f'y_{split}.npy', allow_pickle=False)
        if x.dtype != np.float32 or x.shape != (len(frame), 6, 5000) or not np.isfinite(x).all():
            raise ValueError('Invalid six-lead signals')
        if y.shape != (len(frame), 2) or not np.array_equal(y, frame[LABELS].to_numpy()):
            raise ValueError('Label alignment failure')
        data[split] = TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.float32)))
        counts[split] = dict(recordings=len(frame), patients=frame.patient_id.nunique(),
            **{label: dict(positive=int(y[:, i].sum()), negative=int(len(y)-y[:, i].sum())) for i, label in enumerate(LABELS)})
        for filename in [f'X_{split}.npy', f'y_{split}.npy', f'{split}.csv']:
            fingerprints[filename] = sha256(directory / filename)
    fingerprints['contract.json'] = sha256(directory / 'contract.json')
    return data, dict(contract=contract, counts=counts, sha256=fingerprints, patient_overlap=0), frames


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
    # Cross-dataset evaluation must not introduce patients seen in training/validation.
    training_frames = loaded[args.train][2]
    seen = set(pd.concat(training_frames[:2]).patient_id)
    for name, (_, _, frames) in loaded.items():
        if seen & set(frames[2].patient_id):
            raise ValueError(f'Patient leakage into {name} test')
    args.output.mkdir(parents=True, exist_ok=False)
    set_seed(42)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    model = ECGResNet().to(device)
    data = loaded[args.train][0]
    loaders = {s: DataLoader(d, batch_size=32, shuffle=s == 'train') for s, d in data.items()}
    y = data['train'].tensors[1]
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
            truth = datasets['test'].tensors[1].numpy()
            results['tests'][name] = metrics(truth, scores)
            np.savez(args.output / f'{name}_test_predictions.npz', truth=truth, scores=scores)
    (args.output / 'metrics.json').write_text(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--raw-root', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
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
        prepare(args.raw_root, args.output)
    else:
        if args.epochs < 1:
            parser.error('epochs must be positive')
        run(args)

if __name__ == '__main__':
    main()
