"""Audit local PTB-XL exports and an existing run without training or tuning."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)

from ptbxl_data import load_v6_exports, sha256
from resnet_model import ECGDataset, load_checkpoint, make_loaders, predict, training_class_weights

SPLITS = ('train', 'val', 'test')


def overlap_counts(values):
    """Count shared identities for every split pair; never publish patient IDs."""
    return {f'{left}/{right}': len(set(values[left]) & set(values[right]))
            for index, left in enumerate(SPLITS) for right in SPLITS[index + 1:]}


def signal_hashes(signals):
    return [hashlib.sha256(np.ascontiguousarray(row, dtype='<f4').tobytes()).hexdigest()
            for row in signals]


def near_duplicate_scan(signals, threshold=.99):
    """Screen every cross-split pair with full-length centered absolute correlation.

    This catches aligned copies up to polarity, offset and scale. It is a screen,
    not a proof against every time-shifted, filtered or distorted duplicate.
    """
    normalized = {}
    for split, values in signals.items():
        x = values.reshape(len(values), -1).astype(np.float32)
        x -= x.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError('Constant signals require manual duplicate review')
        normalized[split] = x / norms
    result = {}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            maximum, pairs, highest_pair = -1., 0, None
            reference = normalized[left]
            for start in range(0, len(normalized[right]), 64):
                correlation = np.abs(normalized[right][start:start + 64] @ reference.T)
                local = np.unravel_index(np.argmax(correlation), correlation.shape)
                if float(correlation[local]) > maximum:
                    maximum = float(correlation[local])
                    highest_pair = (int(local[1]), start + int(local[0]))
                pairs += int((correlation >= threshold).sum())
            result[f'{left}/{right}'] = {'max_absolute_correlation': maximum,
                                          'pairs_at_or_above_threshold': pairs, '_highest_pair_indices': highest_pair}
    return {'threshold': threshold, 'method': 'all cross-split pairs; full 5000 samples; centered absolute correlation',
            'pairs': result, 'limitation': 'No exhaustive search over arbitrary time shifts or transformations'}


def checkpoint_metrics(model, dataset, device='cpu'):
    """Verify inference does not update weights or BatchNorm statistics."""
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    loader = make_loaders({'test': dataset}, 32)['test']
    y, probability = predict(model, loader, device)
    assert all(torch.equal(before[key], value.detach().cpu()) for key, value in model.state_dict().items())
    predicted = (probability >= .5).astype(int)
    precision, recall, _ = precision_recall_curve(y, probability)
    return y, probability, {
        'accuracy': float(accuracy_score(y, predicted)),
        'precision': float(precision_score(y, predicted, zero_division=0)),
        'recall': float(recall_score(y, predicted, zero_division=0)),
        'f1': float(f1_score(y, predicted, zero_division=0)),
        'auroc': float(roc_auc_score(y, probability)), 'pr_auc': float(auc(recall, precision)),
        'confusion_matrix': confusion_matrix(y, predicted, labels=[0, 1]).tolist()}


def audit(data, raw_root, run):
    """Return aggregate evidence; all comparisons are read-only."""
    arrays, provenance = load_v6_exports(data, raw_root)
    rows = {}
    for split in SPLITS:
        with (data / f'{split}_manifest.csv').open() as stream:
            rows[split] = list(csv.DictReader(stream))
    identifiers = {field: {split: [str(int(float(row[field]))) if field != 'filename_hr'
                                  else row[field] for row in rows[split]] for split in SPLITS}
                   for field in ('patient_id', 'ecg_id', 'filename_hr')}
    signals = {split: arrays[split][0] for split in SPLITS}
    waveform_hashes = {split: signal_hashes(x) for split, x in signals.items()}
    raw_hashes = {split: [sha256((raw_root / row['filename_hr']).with_suffix('.dat'))
                          for row in rows[split]] for split in SPLITS}
    overlap = {field: overlap_counts(values) for field, values in identifiers.items()}
    overlap['exact_V6_samples'] = overlap_counts(waveform_hashes)
    overlap['raw_multilead_dat_content'] = overlap_counts(raw_hashes)
    if any(count for checks in overlap.values() for count in checks.values()):
        raise ValueError(f'Cross-split leakage detected: {overlap}')
    with (raw_root / 'ptbxl_database.csv').open() as stream:
        source_rows = list(csv.DictReader(stream))
    source = {int(row['ecg_id']): row for row in source_rows}
    # Independently check source alignment, including IDs and physical waveform paths.
    for split in SPLITS:
        for row in rows[split]:
            original = source[int(row['ecg_id'])]
            for field in ('patient_id', 'strat_fold'):
                assert int(float(row[field])) == int(float(original[field]))
            for field in ('scp_codes', 'filename_hr'):
                assert row[field] == original[field]
    model, checkpoint = load_checkpoint(run / 'best_model.pt', 'cpu')
    assert checkpoint['epoch'] == 8, 'This audit is for the original epoch-8 experiment'
    assert checkpoint['metadata']['data_provenance']['file_sha256'] == provenance['file_sha256']
    history = json.loads((run / 'history.json').read_text())
    summary = json.loads((run / 'training_summary.json').read_text())
    assert np.argmin(history['val_loss']) + 1 == checkpoint['epoch'] == summary['best_epoch']
    assert checkpoint['val_loss'] == min(history['val_loss'])
    weights = training_class_weights(ECGDataset(*arrays['train'])).tolist()
    np.testing.assert_allclose(weights, checkpoint['training_config']['class_weights'], rtol=0, atol=0)
    y, probability, metrics = checkpoint_metrics(model, ECGDataset(*arrays['test']))
    with np.load(run / 'predictions.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(y, saved['labels'])
        np.testing.assert_allclose(probability, saved['probabilities'], rtol=1e-6, atol=1e-7)
    saved_metrics = json.loads((run / 'metrics.json').read_text())
    for key, value in metrics.items():
        np.testing.assert_allclose(value, saved_metrics[key])
    confounds = {}
    for label in (0, 1):
        selected = [source[int(row['ecg_id'])] for split in SPLITS for row in rows[split]
                    if int(row['label']) == label]
        ages = np.array([float(row['age']) for row in selected if row['age']])
        from collections import Counter
        confounds[str(label)] = {'median_age': float(np.median(ages)),
                                'sex_counts': dict(Counter(row['sex'] for row in selected)),
                                'device_counts': dict(Counter(row['device'] for row in selected))}
    near = near_duplicate_scan(signals)
    # Investigate the strongest pair in each split comparison, without modifying inputs.
    for pair, checks in near['pairs'].items():
        left, right = pair.split('/')
        i, j = checks.pop('_highest_pair_indices')
        selected_signals = [signals[left][i].reshape(-1), signals[right][j].reshape(-1)]
        residuals, low_frequency_fractions = [], []
        for signal in selected_signals:
            spectrum = np.fft.rfft(signal - signal.mean())
            frequency = np.fft.rfftfreq(len(signal), 1 / 500)
            power = np.abs(spectrum) ** 2
            low_frequency_fractions.append(float(power[frequency < .5].sum() / power.sum()))
            spectrum[frequency < .5] = 0
            residuals.append(np.fft.irfft(spectrum, n=len(signal)))
        checks['highest_pair_low_frequency_power_fractions_below_0_5hz'] = low_frequency_fractions
        checks['highest_pair_residual_correlation_audit_only'] = float(np.corrcoef(residuals)[0, 1])
        checks['highest_pair_labels'] = [int(rows[left][i]['label']), int(rows[right][j]['label'])]
        import wfdb
        a = wfdb.rdrecord(str(raw_root / rows[left][i]['filename_hr']))
        b = wfdb.rdrecord(str(raw_root / rows[right][j]['filename_hr']))
        checks['highest_pair_raw_lead_correlations'] = {
            name: float(np.corrcoef(a.p_signal[:, k], b.p_signal[:, b.sig_name.index(name)])[0, 1])
            for k, name in enumerate(a.sig_name)}
    near_pass = not any(pair['pairs_at_or_above_threshold'] for pair in near['pairs'].values())
    return {
        'status': 'technical_checks_pass_with_historical_test_exposure_caveat' if near_pass else 'manual_near_duplicate_review_required',
        'checkpoint_epoch': checkpoint['epoch'], 'epochs_completed': summary['epochs_completed'],
        'checkpoint_sha256': sha256(run / 'best_model.pt'),
        'split_counts': provenance['split_counts'], 'cross_split_overlap': overlap,
        'within_split_exact_V6_duplicate_excess': {split: len(values) - len(set(values))
                                                 for split, values in waveform_hashes.items()},
        'near_duplicate_screen': near,
        'all_inputs_match_original_raw_V6': True, 'source_metadata_alignment_verified': True,
        'input_contract': 'Only float32 V6 amplitudes, (N,1,5000); y separate; no metadata channels',
        'preprocessing': provenance['preprocessing'],
        'normalization': 'No fitted preprocessing normalization. BatchNorm learns training batches only; eval freezes running statistics.',
        'class_weights_training_only': weights, 'class_weights_match_checkpoint': True,
        'checkpoint_is_minimum_validation_loss': True, 'inference_leaves_model_state_unchanged': True,
        'metrics_match_checkpoint_and_saved_predictions': True, 'test_metrics': metrics,
        'score_separation': {'max_normal_score': float(probability[y == 0].max()),
                             'min_CLBBB_score': float(probability[y == 1].min())},
        'majority_normal_accuracy': float((y == 0).mean()),
        'cohort_confound_descriptives': confounds,
        'historical_test_exposure': {
            'pristine_never_viewed_test_set': False,
            'known_exposure': 'Earlier 500 Hz integration smoke run evaluated this test split before the balanced full run; later evaluation repeated it.',
            'full_run_selection': 'Code and history use training-only weights and validation-only early stopping/checkpoint selection; fixed threshold 0.5.',
            'limit': 'No evidence of score-driven tuning in this session, but artifacts cannot prove absence of human adaptation. Repeated inspection weakens a claim of pristine independent final testing.'},
        'limitations': ['Near-duplicate screen is not exhaustive for shifted/filtered copies or incorrect source patient IDs.',
                        'Strict NORM-only versus isolated high-confidence CLBBB is a restricted, easier cohort; not all pathological negatives.',
                        'Test set has only 25 CLBBB recordings from 21 CLBBB patients; recording metrics are not independent patient-level validation.',
                        'Age, sex and acquisition differences can correlate with morphology; metadata are not input features, but waveform confounding is not ruled out.',
                        'No external validation or unseen final dataset; no further tuning against this test set.'],
        'data_file_sha256': provenance['file_sha256']}



def write_review_bundle(result, run, destination):
    """Copy only small aggregate artifacts; never copy ECG arrays or checkpoints."""
    destination.mkdir(parents=True, exist_ok=True)
    if result['checkpoint_epoch'] != 8 or result['status'] != 'technical_checks_pass_with_historical_test_exposure_caveat':
        raise ValueError('Resolve technical findings before preparing a review bundle')
    summary = json.loads((run / 'training_summary.json').read_text())
    history = json.loads((run / 'history.json').read_text())
    assert summary['best_epoch'] == 8 == int(np.argmin(history['val_loss'])) + 1
    saved = json.loads((run / 'metrics.json').read_text())
    metrics = {key: saved[key] for key in ('accuracy', 'precision', 'recall', 'f1', 'auroc',
                                          'pr_auc', 'average_precision', 'threshold', 'confusion_matrix')}
    metrics.update(checkpoint_epoch=8, checkpoint_sha256=result['checkpoint_sha256'],
                   positive_class='Complete LBBB', test_recordings=640, test_positive_recordings=25,
                   false_negatives=metrics['confusion_matrix'][1][0],
                   interpretation='Development-exposed test split; historical independence caveat in README and audit.json')
    experiment = {'checkpoint_epoch': 8, 'epochs_completed': summary['epochs_completed'],
                  'training_config': summary['training_config'], 'seed': 42,
                  'device': 'cpu', 'sampling_rate_hz': 500, 'samples': 5000, 'lead': 'V6',
                  'best_epoch_training_loss': history['train_loss'][7],
                  'best_epoch_validation_loss': history['val_loss'][7],
                  'checkpoint_sha256': result['checkpoint_sha256'],
                  'audited_resnet_source_sha256': sha256(Path(__file__).with_name('resnet_model.py')),
                  'python': sys.version.split()[0], 'platform': platform.platform(),
                  'preprocessing_commit': '77873b1309ec049f204fbbec3d4af7bdb7976911',
                  'checkpoint_excluded': True, 'patient_level_data_excluded': True}
    for name, content in [('audit.json', result), ('metrics.json', metrics),
                          ('history.json', history), ('experiment.json', experiment)]:
        (destination / name).write_text(json.dumps(content, indent=2, allow_nan=False) + '\n')
    for name in ('training_curves.png', 'confusion_matrix.png', 'evaluation.png'):
        if (run / name).stat().st_size > 500_000:
            raise ValueError(f'Unexpectedly large report image: {name}')
        shutil.copyfile(run / name, destination / name)
    documentation = r"""# Epoch-8 ResNet results — review bundle with qualifications

Technical leakage checks passed; this is **not an unconditional independent-test
certification**. Fold 10 was evaluated during earlier 500 Hz integration smoke
runs before the balanced full experiment, and was evaluated again afterward.
The full-run code and artifacts show validation-only checkpoint selection and
training-only class weights. There is no evidence in this session of test-score
based tuning, but code and saved artifacts cannot establish the absence of human
adaptation after earlier test exposure. Treat these as development-exposed split
results, and obtain an untouched external/patient-held-out evaluation before
making a final independent-validation claim. Do not tune further on fold 10.

## Verified experiment

Original 500 Hz V6, 5000 samples, mV. Strict-v1: only NORM=100 or CLBBB=100,
optional SR; all other codes (including ILBBB/RBBB) exclude a recording.
Patient folds 1–8 train, 9 validation, 10 test; no resplitting or normalization.
Train: 5278 recordings/4994 patients (5095 Normal, 183 CLBBB).
Validation: 614 recordings/593 patients (595 Normal, 19 CLBBB).
Test: 640 recordings/593 patients (615 Normal, 25 CLBBB from 21 CLBBB patients).
Repeated visits within splits are retained; metrics are recording-level.

ResNet architecture unchanged. Adam lr=0.001, batch size=32, seed=42, CPU,
max epochs=30, patience=5, min_delta=0.0001. Class weights [0.51795876,
14.42076492] use training counts only and are fixed for validation. Stopping and
selection use weighted validation cross entropy. Training stopped at epoch 13;
checkpoint epoch 8 has the lowest validation loss. Weighted training loss at
that epoch is 0.0066711134; validation loss is 0.0019281085. The threshold was
fixed at 0.5. Epoch-8 weights and Adam state remain local, excluded from Git;
there is no epoch-13 resume checkpoint or saved random-generator state.

| Test metric | Value |
| --- | ---: |
| Accuracy | 0.996875 |
| CLBBB precision | 0.925925926 |
| CLBBB recall | 1.0 |
| CLBBB F1 | 0.961538462 |
| ROC-AUC | 1.0 |
| PR-AUC (trapezoidal) | 1.0 |
| False negatives | 0 / 25 |

Confusion matrix (true rows, predicted columns, Normal then CLBBB):
`[[613, 2], [0, 25]]`. Metrics were recomputed from the restored epoch-8
checkpoint and checked against saved predictions. The checkpoint SHA-256 is in
`experiment.json` and `metrics.json`; no checkpoint is included here.

## Audit findings and performance investigation

- Zero cross-split shared patient IDs, ECG IDs, high-resolution paths, exact V6
  sample hashes, or raw multilead DAT hashes. No within-split exact V6 copies.
- Every exported input exactly matches original physical V6 samples after
  float32 conversion. Labels, SCP codes, filenames, folds, IDs, age, sex and
  device are not model inputs. Labels enter only the loss or metric calculation.
- No fitted preprocessing normalization. BatchNorm running statistics learn
  training batches only and remain unchanged during validation/inference.
- Unit tests guard test-loader isolation, train-derived weights, label-independent
  inference and frozen BatchNorm. History minimum and checkpoint identify epoch 8.
- Every cross-split pair was screened using full-length centered absolute
  correlation, including polarity/scale/offset invariance. No pair reached 0.99.
  Maximum train/test correlation was 0.9801. Its records have different patient
  and ECG IDs; both are Normal. Approximately 84% and 89% of their V6 power is
  below 0.5 Hz, and most other leads have near-zero cross-record correlation.
  Private visual inspection showed different beat timing with similar slow
  baseline transients. This supports an acquisition/baseline-artifact explanation,
  not a confirmed duplicate. Residual correlation after excluding below 0.5 Hz
  remains 0.9043: transformed duplication cannot be categorically ruled out.
  This audit-only frequency analysis never changed training inputs or weights.
- The screen does not exhaustively test arbitrary shifts, filters, transformed
  copies or errors in source patient identity. Shared acquisition artifacts could
  still provide shortcuts; external validation and signal-quality review remain
  necessary. No cohort or model changes were made based on this investigation.
- Always predicting Normal would give 96.09% accuracy. The cohort is narrowly
  selected, high-confidence Normal versus isolated CLBBB, not CLBBB versus every
  other pathology. Median recording age is 52 for Normal and 68 for CLBBB;
  device distributions also differ (see audit.json). This can make discrimination
  easier and permit waveform confounding even though metadata are excluded.
- Perfect AUC is arithmetically consistent: max Normal score=0.8073271 is below
  min CLBBB score=0.8939498. Two Normal scores exceed threshold 0.5, explaining
  imperfect precision despite perfect ranking. Only 25 positive test recordings
  and 21 positive patients support the result; this is not external validation.

## Files and reproduction

`training_curves.png` shows all 13 epochs, not 13 separately selected models.
`confusion_matrix.png` and `evaluation.png` use the selected epoch-8 model.
`history.json` contains only aggregate epoch metrics. `audit.json` contains
aggregate counts, checks and file fingerprints, with no patient IDs, ECG IDs,
raw waveforms, per-record predictions, local absolute data paths or weights.

From the repository root, with Python 3.11 on a compatible Apple Silicon Mac:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproducible.txt
python -m pytest -q
# Existing local data only; choose a NEW external output directory.
python prepare_ptbxl.py --raw-root "$HOME/Desktop/PTB-XL" \
  --output "$HOME/Desktop/PTB-XL-ResNet-500Hz-reproduction"
# No test-based retuning. Exact results can vary by platform/library version.
OMP_NUM_THREADS=1 python -u resnet_model.py \
  --data "$HOME/Desktop/PTB-XL-ResNet-500Hz-reproduction/v6" \
  --raw-root "$HOME/Desktop/PTB-XL" --device cpu --epochs 30 \
  --patience 5 --min-delta 0.0001 --batch-size 32 --learning-rate 0.001 \
  --seed 42 --output outputs/reproduction_500hz
# For this ORIGINAL experiment, local epoch-8 checkpoint and saved outputs:
OMP_NUM_THREADS=1 python leakage_audit.py \
  --data "$HOME/Desktop/PTB-XL-ResNet-500Hz-processed/v6" \
  --raw-root "$HOME/Desktop/PTB-XL" --run outputs/real_500hz_full_20261109 \
  --output outputs/leakage_audit/audit.json
```

`leakage_audit.py` intentionally asserts epoch 8: it verifies the recorded
experiment, not an arbitrary future run that may select another epoch. To
regenerate this aggregate review bundle from the original local artifacts, add
`--bundle reports/ptbxl_resnet_500hz` to that audit command. Reproduction training
is documented, not rerun during this audit. The original checkpoint is required
to reproduce its inference exactly; a fresh clone contains code and aggregate
results only. The pinned preprocessing Git object must be available; fetch
Prisha's branch if needed, without checking it out or merging it.
"""
    (destination / 'README.md').write_text(documentation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/leakage_audit/audit.json'))
    parser.add_argument('--bundle', type=Path, help='Prepare a small aggregate review bundle with explicit caveats')
    args = parser.parse_args()
    result = audit(args.data, args.raw_root, args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    if args.bundle:
        write_review_bundle(result, args.run, args.bundle)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
