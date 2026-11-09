# Research-001-Codebase

## V6 ResNet baseline at 500 Hz

**Evaluating the Stability of Neural Network-Identified ECG Features** uses
original PTB-XL `records500` recordings: V6, 500 Hz, 10 seconds, **5000 samples**.
Normal is class 0; complete LBBB (`CLBBB`) is class 1. No waveform is downsampled,
resampled, filtered, padded, truncated, or normalized by this pipeline. Synthetic
signals exercise software only. Feature attribution and stability experiments
remain future work.

### Installation and tests

Use Python 3.10 or newer (validated with Python 3.11):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m pip check
```

### Prepare existing local 500 Hz recordings

Keep raw and processed ECG data outside the repository. The local source is
`~/Desktop/PTB-XL`, containing `ptbxl_database.csv`, `scp_statements.csv` and
`records500/`. Use a new output directory:

```bash
python prepare_ptbxl.py --raw-root "$HOME/Desktop/PTB-XL" \
  --output "$HOME/Desktop/PTB-XL-ResNet-500Hz-processed"
```

This runner reads a pinned snapshot of Prisha's code at commit
`77873b1309ec049f204fbbec3d4af7bdb7976911` on
`origin/feature/ptbxl-data-processing`. It executes a temporary copy without
checking out, merging, or modifying her branch or files. If this Git object is
missing in a fresh clone, first run
`git fetch origin feature/ptbxl-data-processing`.

Her metadata inspection, strict label selection and patient/fold checks are
unchanged. Her original extractor supports only `filename_lr`, `records100`,
100 Hz and 1000 samples. The temporary extraction copy is adapted to
**`filename_hr`, `records500`, 500 Hz and 5000 samples**. It selects V6 by channel
name, checks mV units and finite samples, and saves all original samples as
float32. There is no sample-rate conversion. Metadata still contains both path
columns; only `filename_hr` supplies training waveforms. The unchanged metadata
report's `waveform_headers_present` field counts low-resolution headers and is
not evidence of 500 Hz extraction; use raw waveform verification described below.

The runner requires existing local files, performs no downloads, never changes
raw recordings, and refuses an existing output directory. Outputs are
`metadata/`, `v6/`, and `pipeline_source.json`, which records the exact adaptation.
Previous 100 Hz outputs remain separate and are rejected by the current loader.

### Train and evaluate

```bash
# Full real-data run, including every-row comparison to original 500 Hz V6.
python resnet_model.py \
  --data "$HOME/Desktop/PTB-XL-ResNet-500Hz-processed/v6" \
  --raw-root "$HOME/Desktop/PTB-XL" --epochs 30 --output outputs/real_500hz

# Evaluate a saved 500 Hz checkpoint without retraining.
python resnet_model.py \
  --data "$HOME/Desktop/PTB-XL-ResNet-500Hz-processed/v6" \
  --raw-root "$HOME/Desktop/PTB-XL" \
  --checkpoint outputs/real_500hz/best_model.pt \
  --output outputs/real_500hz_evaluation

# Explicit synthetic software test, also using 5000 samples at 500 Hz.
python resnet_model.py --synthetic --epochs 2 --output outputs/synthetic_500hz
```

Real data is the default; without `--data` it looks for `data/processed/v6_500hz`.
Missing real files fail clearly, without falling back to synthetic signals.
Default outputs are `outputs/real_500hz` and `outputs/synthetic_500hz`.
CUDA, Apple Silicon MPS and CPU are selected in that order; override with
`--device cpu`, `cuda` or `mps`. Seed defaults to 42 (`--seed` to change it).
Deterministic operations are requested, but exact results across devices or
PyTorch versions are not guaranteed. Use CPU if a backend reports unsupported
deterministic operations.

The architecture retains its original residual blocks, batch normalization,
ReLU, skip connections, convolution strides, global average pooling and two
output logits. Inputs are now `(N, 1, 5000)`; the convolution strides reduce
internal feature lengths, while the input ECG remains at its original 500 Hz.
CrossEntropyLoss and Adam are unchanged. Labels remain int64; do not cast them
to float. Softmax class 1 is the CLBBB score; predictions use threshold 0.5.
The wider input does not require changing the global pooling classifier.
The full experiment uses training-derived weights and validation-based selection.
The test split was viewed during earlier smoke runs; historical test independence
cannot be certified. See the qualified audit below before interpreting results.

### Data contract and leakage checks

`ptbxl_data.py` reads Prisha's directory export directly:

| Split | Signals | Labels | Row-aligned metadata |
| --- | --- | --- | --- |
| Train | `X_train.npy` | `y_train.npy` | `train_manifest.csv` |
| Validation | `X_val.npy` | `y_val.npy` | `val_manifest.csv` |
| Test | `X_test.npy` | `y_test.npy` | `test_manifest.csv` |

`preprocessing.json` must declare V6, fs=500, samples=5000, mV, float32 signals,
int64 labels, no normalization, and the selection manifest SHA-256. Directory
signals must have shape `(N, 1, 5000)` and labels `(N,)`, with both classes in
each split. Manifests require `ecg_id`, `patient_id`, `scp_codes`, `strat_fold`,
`filename_hr`, `label` and `split`.

Prisha's `strict-v1` policy requires exactly one target (`CLBBB` or `NORM`),
confidence 100 and no additional statement except optional `SR`. Incomplete LBBB,
conflicting targets, uncertain targets and other statements are excluded.
The adapter independently checks labels against this unchanged policy.

Original folds 1–8 remain train, 9 validation, 10 test. The adapter rejects
missing IDs, invalid folds, any patient in multiple folds, duplicate ECG IDs or
waveform paths, label/manifest disagreement, wrong shapes, nonfinite samples,
and unexpected lead/rate/preprocessing. No resplitting or learned normalization
occurs. If normalization is added later, fit it on training data only.

`--raw-root` independently compares **every exported row** with its original
`filename_hr` WFDB recording, checking the header's 500 Hz, 5000 samples, unique
V6 channel, mV units and exact float32 sample agreement. Without this argument,
lead/rate verification relies on exported preprocessing metadata. Official raw
file checksums are not independently revalidated by this integration.

A legacy NPZ interface remains available with `--data archive.npz` and keys
`X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test`, plus
`patient_ids_train`, `patient_ids_val`, `patient_ids_test`. It accepts signals
shaped `(N, 5000)` or `(N, 1, 5000)` and checks finite values, binary labels and
patient separation, but has no manifests or raw-header verification. Prefer the
directory adapter for real experiments.

### Artifacts and model selection

Each run saves `best_model.pt` (lowest validation loss), `history.json`,
`training_curves.png`, `run_metadata.json`, `metrics.json`, `evaluation.png` and
`predictions.npz`. Checkpoints include model/optimizer state, epoch, validation
loss, 500 Hz/5000-sample input configuration and data provenance. Old 100 Hz
checkpoints are rejected. Export fingerprints prevent evaluating a checkpoint
against changed exports. Outputs and model files are Git-ignored.

The best validation checkpoint is restored before test evaluation. Metrics are
accuracy, precision, recall, F1, confusion matrix (true rows, predicted columns,
class order 0,1), AUROC, trapezoidal PR-AUC and separately average precision.
Single-class test AUROC and PR-AUC are null; undefined precision/recall/F1 use
zero. Test arrays are checked for integrity during loading, but never used for
optimization or checkpoint selection. Do not tune using the resulting test scores.

### Local cohort

The 500 Hz extraction retains Prisha's cohort and split counts:

| Split | Normal | CLBBB | Recordings | Unique patients |
| --- | ---: | ---: | ---: | ---: |
| Train (folds 1–8) | 5095 | 183 | 5278 | 4994 |
| Validation (fold 9) | 595 | 19 | 614 | 593 |
| Test (fold 10) | 615 | 25 | 640 | 593 |

Review this conservative cohort policy and imbalance with the research team
before interpreting results. Short integration runs establish software execution,
not clinical validity or feature stability. Full CPU training is complete;
CUDA/MPS execution, external validation and feature-stability experiments remain untested. WFDB currently emits pandas deprecation warnings
which do not prevent the tests from passing.

### 500 Hz integration validation

All 6,532 selected 500 Hz exports were independently compared with their original
V6 waveforms and matched exactly after float32 conversion. All headers satisfied
500 Hz, 5000 samples and mV units. No patient overlap was found. The full
one-epoch CPU run completed training, validation-based checkpoint selection,
checkpoint reload and held-out evaluation. Local ignored artifacts are in
`outputs/real_500hz_integration_smoke/`.

The 51 automated tests pass, including forward/backward passes on 5000 samples,
training, predictions, checkpoint restoration, original-rate V6 verification,
label/fold validation, and rejection of 100 Hz inputs, paths and checkpoints.
Prisha's five original metadata tests also pass unchanged; the new extractor
test verifies her adapted extraction at 500 Hz and rejects 100 Hz. Dependency
consistency checks passed. ECG data, checkpoints and temporary generated outputs
are excluded from Git. Only the small aggregate review bundle under `reports/`
is prepared for version control.

### Balanced training and early stopping

Training now defaults to class-weighted cross entropy. Weights are computed only
from training labels as `N_train / (2 * class_count)`; validation uses the same
fixed weights. Epoch loss is the sum of weighted losses divided by the sum of
sample weights, so its aggregation does not depend on validation batch sizes.
`--unweighted` disables balancing for explicit comparison experiments.

Early stopping monitors validation weighted loss with `--patience 5` and
`--min-delta 0.0001` by default. The checkpoint always preserves the absolute
lowest validation loss; patience resets after an improvement larger than
`min_delta`. A full run can use `--epochs 30` as the maximum. Training history
and curves include class-1 recall and F1 alongside loss and accuracy.
`training_config.json` records class weights, optimizer settings and stopping
configuration; `training_summary.json` records the best epoch and stopping epoch.
The decision threshold stays at 0.5, and test data never influences stopping,
class weighting or checkpoint selection. Weighted losses should not be compared
directly with earlier unweighted-loss runs.


### Qualified leakage audit and epoch-8 review results

See [the experiment and leakage report](reports/ptbxl_resnet_500hz/README.md)
for exact reproduction commands, selection settings, limitations and the result
bundle. This bundle is ready for review, **not an unconditional leakage-free or
independent-test certification**: earlier integration runs evaluated fold 10
before the full balanced run. The code guards against test-driven optimization
and selection, but cannot retrospectively establish absence of human adaptation.
No further tuning should use this exposed test split; external, untouched data
are needed for a stronger independent-validation claim.

The audit found zero cross-split patient, ECG ID, path, exact V6 waveform or raw
multilead DAT overlap. All samples match raw V6, and the input contains no target
metadata. No pair met the all-pair centered-correlation screen at 0.99. The
strongest train/test pair (0.9801) has different beat timing and similar slow
baseline artifacts; transformed copies and acquisition shortcuts are not ruled
out categorically. Class/device/age imbalance and the strict selected cohort
are additional generalization risks. The audit changes neither the cohort nor
the checkpoint and does not retrain.

```bash
OMP_NUM_THREADS=1 python leakage_audit.py \
  --data "$HOME/Desktop/PTB-XL-ResNet-500Hz-processed/v6" \
  --raw-root "$HOME/Desktop/PTB-XL" --run outputs/real_500hz_full_20261109 \
  --output outputs/leakage_audit/audit.json
```

Add `--bundle reports/ptbxl_resnet_500hz` to regenerate only the aggregate review
artifacts after technical checks pass with the documented historical caveat.
The audit is deliberately specific to the recorded epoch-8 checkpoint; it
rejects a different selected epoch. A fresh clone includes no trained weights:
reproduce training with the pinned runtime versions in
`requirements-reproducible.txt`, or supply the original checkpoint locally.
Expected full-run settings are CPU, seed 42, batch 32, lr 0.001, maximum 30
epochs, patience 5 and minimum improvement 0.0001; original training stopped at
13 and selected 8. Exact results across hardware/library versions can differ.

The review bundle includes aggregate metrics, 13-epoch history, audit evidence,
configuration and small plots. It excludes patient IDs, ECG IDs, per-record
predictions, waveforms, checkpoints, absolute local data paths and temporary
inspection plots. Model input/training logic is unchanged by this audit; tests
add guards for duplicate detection, metadata-independent predictions,
validation BatchNorm freezing and training-only class weighting.
