# Six-Lead PTB-XL ECG Modeling

This project trains one-dimensional ResNet classifiers for complete and incomplete
left bundle branch block using six PTB-XL ECG leads: **II, III, aVR, aVF, V3, V4**.
Inputs are native **500 Hz** recordings with **5,000 samples per lead** (10 seconds).

## Experiment summary

| Experiment | Dataset | Leads, in input order | Input shape per ECG | Targets | Status |
|---|---|---|---|---|---|
| Experiment 5 | Dataset 1: LBBB vs. Normal | II, III, aVR, aVF, V3, V4 | `(6,5000)` at 500 Hz | Independent CLBBB and ILBBB | Training and Dataset 1 test evaluation completed; Dataset 2 evaluation pending |
| Experiment 6 | Dataset 2: LBBB vs. All Others | II, III, aVR, aVF, V3, V4 | Required: `(6,5000)` at 500 Hz | Planned independent CLBBB and ILBBB | Pending: dataset policy, exports, training and evaluation |

## Shared data handling

The source files are `ptbxl_database.csv`, `scp_statements.csv`, and native WFDB
recordings in `records500/`, selected using **`filename_hr`**. Signals are stored
as float32 physical amplitudes in mV, with channels before time. No filtering,
resampling, upsampling, padding, truncation, augmentation or fitted amplitude
normalization is applied.

Official PTB-XL folds are retained: **1–8 for training, 9 for validation, and
10 for testing**. There is no random resplitting. Repeated recordings within a
patient's assigned split remain included. The completed Dataset 1 cohort has zero
cross-split patient overlap. Loaders validate identifiers, folds, shapes,
finite samples and label alignment, and reject duplicate recordings or paths.
BatchNorm learns statistics during training and freezes them for evaluation.

## Experiment 5: six leads at 500 Hz

### Dataset 1 and labels

A recording is selected if it contains CLBBB, ILBBB, or qualifies as Normal.
The targets are separate binary values, ordered **`[CLBBB,ILBBB]`**, determined
by SCP code presence without an additional confidence threshold. LBBB-positive
recordings may have other diagnoses and may have both labels.

Normal recordings must contain `NORM` and no other code marked `diagnostic == 1`
in `scp_statements.csv`. Non-diagnostic rhythm/form statements are permitted.
Normal has targets `[0,0]`.

| Split | Recordings | Unique patients | CLBBB positive / negative | ILBBB positive / negative |
|---|---:|---:|---:|---:|
| Train | 7,733 | 7,174 | 428 / 7,305 | 62 / 7,671 |
| Validation | 975 | 916 | 54 / 921 | 7 / 968 |
| Test | 974 | 883 | 54 / 920 | 8 / 966 |

Each negative count refers to that output independently. The exact lead order
is **II, III, aVR, aVF, V3, V4**. Headers are matched case-insensitively.
The extractor requires raw shape `(5000,12)` at 500 Hz and saves
**`(N,6,5000)`**. All 9,682 exports were independently reread from native WFDB
files and matched every selected float32 sample exactly.

### ResNet architecture

`ECGResNet` in `learning_pipeline/07_resnet_model.py` has:

- A kernel-7, stride-1 stem convolution from **6 to 32** channels, BatchNorm and ReLU.
- Three residual blocks with widths **32→32, 32→64, 64→128** and strides **1,2,2**.
  Each block contains two kernel-7 convolutions with BatchNorm, an identity or
  kernel-1 projection shortcut, and ReLU after the first convolution and addition.
- Adaptive average pooling over time and a **128→2** linear classifier.
  Sigmoid maps the logits to independent CLBBB/ILBBB probabilities.

The six-input model has **243,234 trainable parameters**. Adaptive pooling
supports 5,000-sample recordings. Input **`(batch_size,6,5000)`** produces
**`(batch_size,2)`**, with forward/backward execution covered by tests.

### Training and model selection

| Training setting | Final 500 Hz run |
|---|---|
| Optimizer / learning rate | Adam / 0.001 |
| Seed / batch size / device | 42 / 32 / Apple Silicon MPS |
| Loss | `BCEWithLogitsLoss` with separate positive weights |
| Positive weights | Training negatives / positives: CLBBB 17.06775665; ILBBB 123.72580719 |
| Maximum epochs / early stopping | 30 / patience 7, monitoring validation weighted BCE |
| Checkpoint | Lowest validation loss; restored before test inference |
| Completed epochs / best epoch | **10 / 3** |
| Best validation loss | **0.4496812247** |
| Main threshold | **0.5** for each sigmoid output |

Training starts from fresh initialization. Positive weights use training labels
only and stay fixed for validation. Training batches are shuffled, validation controls model selection, and
PyTorch deterministic operations are requested. Test scores do not control the implemented optimization, stopping,
weights or thresholds. Experiment 5 has no additional-seed study,
cross-validation, architecture comparison or validation threshold sweep.

### Final 500 Hz test results

The evaluator reports **accuracy, precision, recall, F1 and confusion matrices
separately for each label**, using probability ≥ 0.5. It does not calculate
ROC-AUC or average precision. Undefined precision/recall/F1 use zero.

| Label | Accuracy | Precision | Recall | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CLBBB | **97.54%** | **70.27%** | **96.30%** | **81.25%** | 898 | 22 | 2 | 52 |
| ILBBB | **86.55%** | **5.11%** | **87.50%** | **9.66%** | 836 | 130 | 1 | 7 |

Confusion matrices use true rows / predicted columns, negative then positive:
CLBBB **`[[898,22],[2,52]]`**; ILBBB **`[[836,130],[1,7]]`**.

These results are from the **final native 500 Hz run**, with its epoch-3 weights.
CLBBB detects 52/54 positives. ILBBB detects 7/8 positives but produces 130 false
positives, yielding low precision and F1. No earlier 100 Hz results are presented
as 500 Hz results.

## Experiment 6 and Dataset 2: pending

Dataset 2 is intended for **LBBB vs. All Others**. Its exact selection policy,
including ambiguous-code handling and negative definitions, and validated
exports are unavailable. Dataset 1 is not substituted. No counts, trained
weights or performance are available for Experiment 6.

The six-lead runner supports a future approved Dataset 2 export using 500 Hz,
5,000 samples, the same ordered six leads, separate `[CLBBB,ILBBB]` targets,
and official patient-separated folds. It requires `contract.json`,
`train.csv`, `val.csv`, `test.csv`, and corresponding `X_*.npy` / `y_*.npy` files.
Arrays must have shapes `(N,6,5000)` and `(N,2)`. Manifests require `ecg_id`,
`patient_id`, `strat_fold`, `filename_hr`, `CLBBB` and `ILBBB`.

The contract must declare `dataset=dataset2`, ordered `leads` and `labels`,
`fs=500`, `samples=5000`, `source_column=filename_hr`,
`waveform_directory=records500`, `units=mV`, `normalization=none`, and a nonempty
label-policy description. Export consistency checks do not establish whether
an external builder correctly implements the approved selection policy.

| Model | Dataset 1 test | Dataset 2 test |
|---|---|---|
| Experiment 5 | Completed | **Pending: dataset unavailable** |
| Experiment 6 | **Pending: model not trained** | **Pending: dataset and model unavailable** |

## Reproduction

### Environment and input files

The completed Experiment 5 run used Python **3.11.9** and PyTorch **2.15.0** on macOS arm64.
Full package pins are in `requirements-reproducible.txt`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproducible.txt
python -m pip check
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ecg-mpl python -m pytest -q
```

Provide local PTB-XL metadata and `records500` WFDB `.hea`/`.dat` files; the
runners do not download data. Replace the example raw-data location below with
your own. Choose new output directories to preserve existing artifacts.

### Experiment 5

```bash
python non_diagnostic_leads.py prepare \
  --raw-root "$HOME/Desktop/PTB-XL" --output data/experiment5_500hz
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ecg-mpl python non_diagnostic_leads.py train \
  --dataset1 data/experiment5_500hz --epochs 30 --device mps \
  --output outputs/experiment5_500hz
```

Use `--device cpu` when MPS is unavailable; CPU is the six-lead CLI default.
Preparation and training refuse existing output directories. The loader requires
`contract.json` and the split CSV/array files. Training restores the lowest-loss
checkpoint and evaluates Dataset 1 test automatically.

To evaluate a checkpoint without retraining, use its matching exports:

```bash
OMP_NUM_THREADS=1 python non_diagnostic_leads.py train \
  --dataset1 data/experiment5_500hz --device mps \
  --checkpoint outputs/experiment5_500hz/best_model.pt \
  --output outputs/experiment5_500hz_evaluation
```

The paths above create reproduction runs. A new training run may select a
different epoch or score; exact results across devices/library versions are
not guaranteed. Exact inference for a saved run requires that run's original
checkpoint and matching export fingerprints.

### Future Dataset 2 runs

Run these **only after the approved Dataset 2 export is available**. There is
no Dataset 2 preparation command implemented here.

```bash
# Train Experiment 6 and evaluate its best checkpoint on both test cohorts.
OMP_NUM_THREADS=1 python non_diagnostic_leads.py train \
  --dataset1 data/experiment5_500hz --dataset2 data/dataset2_500hz \
  --train dataset2 --device mps --output outputs/experiment6_500hz

# Evaluate an Experiment 5 checkpoint on both test cohorts without retraining.
OMP_NUM_THREADS=1 python non_diagnostic_leads.py train \
  --dataset1 data/experiment5_500hz --dataset2 data/dataset2_500hz \
  --checkpoint outputs/experiment5_500hz/best_model.pt --device mps \
  --output outputs/experiment5_both_tests
```

The runner rejects test patients present in that model's training or validation
cohort, including across datasets, without silently resplitting the data.

## Artifacts and verification

The implementation is `non_diagnostic_leads.py`; the model is
`learning_pipeline/07_resnet_model.py`. A training run saves `best_model.pt`,
`history.json`, `training_config.json`, `provenance.json`, `metrics.json`, and
per-dataset test predictions in NPZ format with truths and probability scores.
The checkpoint records model state, selected epoch/loss, seed, positive weights,
lead/label order, training configuration, architecture hash and export provenance.
Evaluation-only mode verifies the training dataset identity, export fingerprints,
architecture hash and ordered leads/labels before inference. The checkpoint does
not include all state required for exact interrupted-training resumption.

Small aggregate results and raw-input verification evidence are under `reports/`.
The completed Experiment 5 epoch-3 checkpoint is identified by SHA-256:

```text
92ae4d18f654a5ab9a6a8933735c6e5d6dae7d984c442890131bf74aa6144a35
```

Its history contains 10 epochs, and epoch 3 has the minimum validation loss.
Saved predictions reproduce the reported test metrics. Export contracts and
fingerprints establish the 500 Hz, 5,000-sample input configuration.
The latest full-suite check passed **61 tests**, with existing WFDB/pandas
deprecation warnings. Six-lead checks cover forward/backward execution, labels,
patient separation, sampling frequency, lead order, checkpoint restoration and
evaluation-only execution. Synthetic fixtures are software checks, not results.
Raw data, waveform exports, per-record predictions, model weights and caches
are Git-ignored and absent from a fresh checkout.

## Limitations

- ILBBB has only 62 training, seven validation and eight test positives. One
  test error changes recall by 12.5 percentage points. Accuracy alone is
  insufficient for this imbalanced task.
- Fold 10 has prior development exposure. These are internal held-out results,
  not pristine independent final validation. No further tuning should use it.
- Patient separation and raw-sample agreement do not exclude acquisition
  confounding, transformed duplicates or source patient-identity errors.
  Metrics are recording-level rather than independent patient-level estimates.
- Dataset 2 testing, Experiment 6 training, external validation and feature
  stability analysis remain pending. No clinical deployment validation is claimed.
