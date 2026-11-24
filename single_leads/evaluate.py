"""Score all eight frozen checkpoints on the Dataset 1 and Dataset 2 fold-10 tests.

Reports AUROC and AUPRC (average precision) for CLBBB and ILBBB with record-level bootstrap
95% CIs, plus paired bootstrap differences (median beat minus full ECG) for each lead, using
the same resampled records for both models. Thresholded metrics at 0.50 are supplementary.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from common import LEADS, OUTCOMES, INPUTS
from model import ECGResNet
from train import channel

B, BOOT_SEED = 2000, 2026
POPULATIONS = ['dataset1_test', 'dataset2_test']


def run_dir(root, kind, lead):
    return root / f'{kind}_{lead}'


def score(model, x):
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(model(torch.from_numpy(x[i:i + 64]))).numpy() for i in range(0, len(x), 64)])


def boot_indices(y, n_boot, seed):
    """Resample records with replacement; keep only resamples containing both classes for every outcome."""
    rng = np.random.default_rng(seed); out = []
    while len(out) < n_boot:
        idx = rng.integers(0, len(y), len(y))
        if all(0 < y[idx, j].sum() < len(idx) for j in range(y.shape[1])):
            out.append(idx)
    return np.array(out)


def ci(values):
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def evaluate(data, root):
    torch.set_num_threads(3)
    test = pd.read_csv(data / 'dataset2_test.csv'); y_all = np.load(data / 'y_dataset2_test.npy')
    masks = {'dataset2_test': np.ones(len(test), bool), 'dataset1_test': test.dataset1.to_numpy(bool)}
    scores, provenance = {}, {}
    for kind in INPUTS:
        for lead in LEADS:
            d = run_dir(root, kind, lead)
            assert (d / 'complete.json').exists(), f'{d} has not finished training'
            ckpt = torch.load(d / 'best_model.pt', map_location='cpu', weights_only=True)
            assert ckpt['config']['input'] == kind and ckpt['config']['lead'] == lead
            model = ECGResNet(1); model.load_state_dict(ckpt['model_state_dict']); model.eval()
            s = score(model, channel(data, kind, 'dataset2_test', lead))
            scores[kind, lead] = s
            pred = test[['ecg_id', 'patient_id', 'dataset1', *OUTCOMES]].copy()
            pred[[f'p_{o}' for o in OUTCOMES]] = s
            pred.to_csv(d / 'dataset2_test_predictions.csv', index=False)
            provenance[f'{kind}_{lead}'] = {'best_epoch': ckpt['epoch'], 'val_loss': ckpt['val_loss'],
                                            'checkpoint_sha256': hashlib.sha256((d / 'best_model.pt').read_bytes()).hexdigest()}
    rows, deltas = [], []
    for pop in POPULATIONS:
        mask = masks[pop]; y = y_all[mask]
        idx = boot_indices(y, B, BOOT_SEED)
        for j, outcome in enumerate(OUTCOMES):
            t = y[:, j]
            boot = {}
            for (kind, lead), s in scores.items():
                p = s[mask, j]
                auroc, auprc = roc_auc_score(t, p), average_precision_score(t, p)
                b_roc = np.array([roc_auc_score(t[i], p[i]) for i in idx])
                b_pr = np.array([average_precision_score(t[i], p[i]) for i in idx])
                boot[kind, lead] = (b_roc, b_pr)
                tn, fp, fn, tp = confusion_matrix(t, p >= .5, labels=[0, 1]).ravel()
                rows.append(dict(population=pop, outcome=outcome, input=kind, lead=lead, n=int(len(t)), positives=int(t.sum()),
                                 prevalence=float(t.mean()), auroc=auroc, auroc_ci_low=ci(b_roc)[0], auroc_ci_high=ci(b_roc)[1],
                                 auprc=auprc, auprc_ci_low=ci(b_pr)[0], auprc_ci_high=ci(b_pr)[1],
                                 sensitivity_at_050=tp / (tp + fn), specificity_at_050=tn / (tn + fp),
                                 TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp)))
            for lead in LEADS:
                (fr, fp_), (mr, mp) = boot['full', lead], boot['median', lead]
                full = [r for r in rows if r['population'] == pop and r['outcome'] == outcome and r['lead'] == lead]
                f, m = (next(r for r in full if r['input'] == k) for k in ('full', 'median'))
                deltas.append(dict(population=pop, outcome=outcome, lead=lead,
                                   delta_auroc=m['auroc'] - f['auroc'], delta_auroc_ci_low=ci(mr - fr)[0], delta_auroc_ci_high=ci(mr - fr)[1],
                                   delta_auprc=m['auprc'] - f['auprc'], delta_auprc_ci_low=ci(mp - fp_)[0], delta_auprc_ci_high=ci(mp - fp_)[1]))
    pd.DataFrame(rows).to_csv(root / 'metrics.csv', index=False)
    pd.DataFrame(deltas).to_csv(root / 'median_minus_full.csv', index=False)
    (root / 'evaluation_provenance.json').write_text(json.dumps(
        {'checkpoints': provenance, 'bootstrap_resamples': B, 'bootstrap_seed': BOOT_SEED, 'threshold': 0.5,
         'test_fold': 10, 'dataset1_test_is_subset_of_dataset2_test': True}, indent=2))
    print(pd.DataFrame(rows)[['population', 'outcome', 'input', 'lead', 'auroc', 'auprc']].to_string(index=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True); p.add_argument('--results', type=Path, required=True)
    a = p.parse_args(); evaluate(a.data, a.results)
