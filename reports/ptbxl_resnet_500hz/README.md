# Epoch-8 ResNet results — review bundle with qualifications

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
