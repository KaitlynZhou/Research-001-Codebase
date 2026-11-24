"""Train one single-lead model on Dataset 1 (folds 1-8, fold 9 for selection). Test arrays are never opened.

Settings are held at the six-lead branch's values: seed 42, Adam lr 1e-3, batch 32, at most
30 epochs, early-stopping patience 7, training-derived pos_weight, checkpoint at minimum
validation weighted BCE, no normalization or augmentation.
"""
import argparse, hashlib, json, os, random, time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from common import LEADS, OUTCOMES, INPUTS, SEED
from model import ECGResNet

PATIENCE = 7


def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


def channel(data, kind, split, lead):
    x = np.load(data / f'X_{kind}_{split}.npy', mmap_mode='r')
    return np.ascontiguousarray(x[:, [LEADS.index(lead)], :])


def train(data, out, kind, lead, epochs=30, threads=3):
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'best_model.pt').exists():
        raise FileExistsError(f'{out} already has a checkpoint; use a fresh directory.')
    torch.set_num_threads(threads)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    dev = device()
    model = ECGResNet(1).to(dev)
    xs = {s: channel(data, kind, s, lead) for s in ('train', 'val')}
    ys = {s: np.load(data / f'y_{s}.npy') for s in ('train', 'val')}
    assert xs['train'].shape[2] == INPUTS[kind]['samples']
    loaders = {s: DataLoader(TensorDataset(torch.from_numpy(xs[s]), torch.from_numpy(ys[s])),
                             batch_size=32, shuffle=s == 'train') for s in ('train', 'val')}
    pos = ys['train'].sum(0); neg = len(ys['train']) - pos
    assert ((pos > 0) & (neg > 0)).all()
    weights = torch.tensor(neg / pos, dtype=torch.float32, device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config = dict(input=kind, lead=lead, **INPUTS[kind], outcomes=OUTCOMES, seed=SEED, batch_size=32,
                  max_epochs=epochs, learning_rate=1e-3, patience=PATIENCE, optimizer='Adam', weight_decay=0,
                  pos_weight=weights.cpu().tolist(), selection='minimum validation weighted BCE',
                  normalization='none; physical mV', train='Dataset 1 folds 1-8', validation='Dataset 1 fold 9',
                  n_train=len(ys['train']), n_val=len(ys['val']), device=str(dev), torch=str(torch.__version__),
                  parameters=sum(p.numel() for p in model.parameters()))
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    best, stale, history, start = float('inf'), 0, [], time.time()
    for epoch in range(1, epochs + 1):
        row = {'epoch': epoch}
        for split, loader in loaders.items():
            training = split == 'train'; model.train(training); total = 0.0
            for x, y in loader:
                x, y = x.to(dev), y.to(dev)
                if training: optimizer.zero_grad()
                with torch.set_grad_enabled(training):
                    loss = loss_fn(model(x), y)
                    if training: loss.backward(); optimizer.step()
                total += loss.item() * len(x)
            row[split + '_loss'] = total / len(loader.dataset)
        assert np.isfinite([row['train_loss'], row['val_loss']]).all()
        history.append(row); (out / 'history.json').write_text(json.dumps(history, indent=2))
        print(kind, lead, row, 'elapsed_s', round(time.time() - start), flush=True)
        if row['val_loss'] < best:
            best, stale = row['val_loss'], 0
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'val_loss': best, 'config': config},
                       out / 'best_model.pt')
        else:
            stale += 1
            if stale >= PATIENCE: break
    ckpt = torch.load(out / 'best_model.pt', map_location='cpu', weights_only=True)
    (out / 'complete.json').write_text(json.dumps({
        'epochs_run': epoch, 'best_epoch': ckpt['epoch'], 'seconds': time.time() - start, 'best_validation_loss': best,
        'checkpoint_sha256': hashlib.sha256((out / 'best_model.pt').read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--input', choices=list(INPUTS), required=True); p.add_argument('--lead', choices=LEADS, required=True)
    p.add_argument('--epochs', type=int, default=30); p.add_argument('--threads', type=int, default=3)
    a = p.parse_args(); train(a.data, a.out, a.input, a.lead, a.epochs, a.threads)
