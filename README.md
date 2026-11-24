# Six-Lead PTB-XL ECG Modeling: Experiments 5 and 6

Both experiments classify complete and incomplete left bundle branch block using
native PTB-XL **500 Hz** recordings. Inputs have exactly **5,000 samples per lead**
(10 seconds), ordered **II, III, aVR, aVF, V3, V4**, shape **`(6,5000)`**.
CLBBB and ILBBB are independent binary outputs of the same ResNet architecture.

## Repository structure

- `learning_pipeline/`: preprocessing, model, training/evaluation and source verification scripts.
- `results/`: experiment reports, metrics and verification records.
- `outputs/`: saved model checkpoints (original paths retained).
- `tests/`: automated tests; `docs/`: supplementary documentation.

## Experiment summary

| Experiment | Training cohort | Model | Dataset 1 test | Dataset 2 test |
|---|---|---|---|---|
| Experiment 5 | Dataset 1: LBBB vs. Normal | Verified existing epoch-3 checkpoint reused | Completed | Completed |
| Experiment 6 | Dataset 2: LBBB vs. All Others | Separate model trained from fresh initialization | Completed | Completed |

The existing Dataset 1 export and Experiment 5 model were reused after exact
source verification. Dataset 2 was prepared from its designated builder's logic;
only waveform extraction was changed to the six native 500 Hz channels.

## Dataset sources and verification

| Dataset | Authoritative source | Preparation file | Pinned revision |
|---|---|---|---|
| Dataset 1 | `feature/dataset1-12lead-clean` | `learning_pipeline/02_build_dataset1.py` | `4d82a777d7b811ea35e57d1abfc9cfb5c8128f6e` |
| Dataset 2 | `kevin/experiment2-dataset2-all12` | `experiment2/01_build_dataset2.py` | `3e01b5ebedc1ec2db3f6ac71714d1d86b0f28127` |

`learning_pipeline/verify_dataset_sources.py` executes only the pinned builders' in-memory selection
statements, excluding their output-writing code. It compares actual exports with
those references: record IDs, patient IDs, CLBBB/ILBBB labels, folds, row order and
native high-resolution paths all match. Raw metadata SHA-256 hashes are checked.
See [source verification](results/dataset_sources_verification.json).

### Dataset 1 selection

Include every recording containing CLBBB or ILBBB, allowing additional diagnoses.
Include a Normal negative when `NORM` is present and no other code is marked
`diagnostic == 1` in `scp_statements.csv`. Targets are `[CLBBB,ILBBB]` based on
code presence without an additional confidence cutoff; Normal is `[0,0]`.

**This is the source branch's diagnostic-only Normal rule.** It permits form/rhythm
annotations, including abnormalities not flagged diagnostic. The assignment
preserves that rule exactly; the cohort must not be described as guaranteed
free of every abnormality or disease. No stricter negative filter was introduced.

### Dataset 2 selection

Include **all 21,799 source recordings**. CLBBB/ILBBB code presence gives the two
labels; every recording with neither code is an All Others negative, including
other diagnoses and abnormalities. Positive recordings retain comorbidities.
The source builder also asserts that no recording has both labels; that assertion
passes in this local corpus and is preserved. Model outputs still use independent
sigmoids and binary loss, not a mutually exclusive softmax.

### Counts and official patient splits

Official folds **1–8 train, 9 validate, 10 test** are retained without resplitting.
Repeated recordings within a patient's assigned split remain included.

| Dataset | Split | Recordings | Unique patients | CLBBB positive / negative | ILBBB positive / negative |
|---|---|---:|---:|---:|---:|
| Dataset 1 | Train | 7,733 | 7,174 | 428 / 7,305 | 62 / 7,671 |
| Dataset 1 | Validation | 975 | 916 | 54 / 921 | 7 / 968 |
| Dataset 1 | Test | 974 | 883 | 54 / 920 | 8 / 966 |
| Dataset 2 | Train | 17,418 | 15,023 | 428 / 16,990 | 62 / 17,356 |
| Dataset 2 | Validation | 2,183 | 1,942 | 54 / 2,129 | 7 / 2,176 |
| Dataset 2 | Test | 2,198 | 1,904 | 54 / 2,144 | 8 / 2,190 |

Negative counts refer to each output separately. Dataset 2 has 16,928/2,122/2,136
recordings with neither LBBB label in train/validation/test. Dataset 1 is an exact
same-fold subset of Dataset 2, with matching patient IDs, labels and filenames.

## Native 500 Hz preprocessing and validation

`learning_pipeline/non_diagnostic_leads.py` reads `ptbxl_database.csv`, `scp_statements.csv` and WFDB
recordings from `records500/`, using **`filename_hr`**. It verifies native 500 Hz,
raw shape `(5000,12)`, finite amplitudes, unique header lead names and mV units.
It selects the six channels by case-insensitive header names and saves float32
arrays with shape `(N,6,5000)` and labels `(N,2)`.

There is no resampling, upsampling, filtering, padding, truncation, augmentation
or fitted amplitude normalization. All 9,682 Dataset 1 and 21,799 Dataset 2 exports
were independently compared with their native WFDB samples; every selected
float32 sample matched exactly. Dataset 2 evidence is in
[raw verification](results/dataset2_raw_verification.json).

Waveform writing and loading are disk-backed to bound memory usage. Each batch
copies waveform rows into tensors. `contract.json` declares dataset, ordered
leads/labels, 500 Hz, 5,000 samples, mV, native HR paths, no normalization, and
label policy. Dataset 2 also records its source revision. The loader verifies
contracts, finite inputs, labels aligned to manifests, identifiers, duplicates,
fold membership and patient separation. Invalid data fails rather than being
silently substituted or resplit.

## Shared ResNet architecture

`ecg_resnet.ECGResNet` has **243,234 trainable parameters**:

| Stage | Operation | Output per recording |
|---|---|---|
| Input | Six ordered ECG channels | `(6,5000)` |
| Stem | Conv1D 6→32, kernel 7, padding 3, stride 1; BatchNorm; ReLU | `(32,5000)` |
| Residual block 1 | 32→32, stride 1 | `(32,5000)` |
| Residual block 2 | 32→64, stride 2 | `(64,2500)` |
| Residual block 3 | 64→128, stride 2 | `(128,1250)` |
| Pool | Adaptive average pooling over time | `(128,1)` |
| Classifier | Linear 128→2 | Two logits |
| Probabilities | Independent sigmoid per logit | `[P(CLBBB),P(ILBBB)]` |

Each block contains two kernel-7 convolutions with BatchNorm, an identity or
kernel-1 projected shortcut, and ReLU after the first convolution and shortcut
addition. Convolution biases are enabled. BatchNorm statistics learn during
training and freeze for validation/test. Adaptive pooling supports 5,000 samples.
Both models use this exact architecture and ordered outputs. Verification against
the Dataset 2 source model confirms matching parameter keys/shapes and identical
forward output with six input channels; see
[architecture verification](results/model_architecture_verification.json).

## Training configuration

| Setting | Experiment 5 | Experiment 6 |
|---|---|---|
| Training dataset | Dataset 1 | Dataset 2 |
| Architecture | Shared six-lead ECGResNet | Same |
| Seed / batch size | 42 / 32 | 42 / 32 |
| Optimizer / learning rate | Adam / 0.001 | Adam / 0.001 |
| Device | Apple Silicon MPS | Apple Silicon MPS |
| Loss | Weighted `BCEWithLogitsLoss` | Same loss |
| Positive weight: CLBBB | 17.06775665 | 39.69626236 |
| Positive weight: ILBBB | 123.72580719 | 279.93548584 |
| Maximum epochs / patience | 30 / 7 | 30 / 7 |
| Completed epochs / best epoch | 10 / 3 | 18 / 11 |
| Best validation weighted BCE | 0.4496812247 | 0.5225658352 |
| Main probability threshold | 0.5 per label | 0.5 per label |

For each label, positive weight is training negatives divided by training
positives. The same fixed weights are used for validation. Only training batches
are shuffled. Early stopping counts seven consecutive epochs without a strictly
lower validation loss; the absolute lowest-loss checkpoint is saved and restored
before either test evaluation. Test data never drives weights, optimization,
checkpoint selection, stopping or thresholds.

Experiment 5's saved model was reused without changing its tensor values.
Experiment 6 started from fresh initialization and did not load Experiment 5
weights. Deterministic PyTorch operations are requested. No additional seeds,
cross-validation, architecture experiments or threshold tuning were run.

## All four held-out evaluations

Both models use the same two test exports. Accuracy, precision, recall, F1 and
confusion matrices are computed **separately for CLBBB and ILBBB**, using sigmoid
probability **≥ 0.5**. Undefined precision/recall/F1 use zero. ROC-AUC and average
precision are not computed by this runner. All metrics are recording-level.

| Model | Test cohort | Label | Accuracy | Precision | Recall | F1 | TN | FP | FN | TP |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Experiment 5 | Dataset 1 | CLBBB | 97.54% | 70.27% | 96.30% | 81.25% | 898 | 22 | 2 | 52 |
| Experiment 5 | Dataset 1 | ILBBB | 86.55% | 5.11% | 87.50% | 9.66% | 836 | 130 | 1 | 7 |
| Experiment 5 | Dataset 2 | CLBBB | 88.26% | 16.88% | 96.30% | 28.73% | 1888 | 256 | 2 | 52 |
| Experiment 5 | Dataset 2 | ILBBB | 68.56% | 1.00% | 87.50% | 1.99% | 1500 | 690 | 1 | 7 |
| Experiment 6 | Dataset 1 | CLBBB | 98.77% | 81.82% | 100.00% | 90.00% | 908 | 12 | 0 | 54 |
| Experiment 6 | Dataset 1 | ILBBB | 88.71% | 6.03% | 87.50% | 11.29% | 857 | 109 | 1 | 7 |
| Experiment 6 | Dataset 2 | CLBBB | 94.54% | 31.03% | 100.00% | 47.37% | 2024 | 120 | 0 | 54 |
| Experiment 6 | Dataset 2 | ILBBB | 79.62% | 1.54% | 87.50% | 3.03% | 1743 | 447 | 1 | 7 |

TN/FP/FN/TP define each confusion matrix as `[[TN,FP],[FN,TP]]`, true rows and
predicted columns, negative then positive. Exact values and saved-score checks
are in [the completed assignment report](results/assignment_results.json).

Experiment 5 retains CLBBB recall 96.30% on Dataset 2, but its CLBBB precision is 16.88% and ILBBB precision is 1.00%. Its Dataset 2 false positives are 256 CLBBB and 690 ILBBB.

Experiment 6's F1 on Dataset 1 / Dataset 2 is 90.00% / 47.37% for CLBBB and 11.29% / 3.03% for ILBBB. All results use the prespecified 0.5 threshold; no test-driven retuning was performed.

### Leakage controls

Each dataset has zero patient overlap between train/validation/test. Before
cross-dataset inference, every test patient is checked against that model's
training **and validation** patient sets: all four overlaps are zero. Shared
recordings must have identical patient IDs, labels, official folds and HR paths.
Predictions for the Dataset 1 subset are checked against the same records in
Dataset 2. The test cohorts overlap and are **not independent external datasets**.

## Reproduction

Recorded runs used Python **3.11.9**, PyTorch **2.15.0**, and macOS arm64/MPS.
Direct dependency pins are in `requirements-reproducible.txt`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproducible.txt
python -m pip check
OMP_NUM_THREADS=1 python -m pytest -q -p no:cacheprovider
```

Provide local `ptbxl_database.csv`, `scp_statements.csv` and native `records500`
WFDB pairs. No downloads are performed. Use new output paths; existing exports,
models and reports are not overwritten. Replace the example raw-data path below.

### Prepare both datasets

```bash
python learning_pipeline/non_diagnostic_leads.py prepare --dataset dataset1 \
  --raw-root "$HOME/Desktop/PTB-XL" --output data/dataset1_500hz_reproduction
python learning_pipeline/non_diagnostic_leads.py prepare --dataset dataset2 \
  --raw-root "$HOME/Desktop/PTB-XL" --output data/dataset2_500hz_reproduction
```

For source verification, the two pinned source revisions must be available in
local Git objects. If needed, fetch their designated branches without merging:

```bash
git fetch --no-tags origin feature/dataset1-12lead-clean kevin/experiment2-dataset2-all12
python learning_pipeline/verify_dataset_sources.py --raw-root "$HOME/Desktop/PTB-XL" \
  --dataset1 data/dataset1_500hz_reproduction --dataset2 data/dataset2_500hz_reproduction \
  --output results/dataset_sources_reproduction.json
```

### Train each model and evaluate both test datasets

```bash
OMP_NUM_THREADS=1 python learning_pipeline/non_diagnostic_leads.py train \
  --dataset1 data/dataset1_500hz_reproduction --dataset2 data/dataset2_500hz_reproduction \
  --train dataset1 --epochs 30 --device mps --output outputs/experiment5_reproduction
OMP_NUM_THREADS=1 python learning_pipeline/non_diagnostic_leads.py train \
  --dataset1 data/dataset1_500hz_reproduction --dataset2 data/dataset2_500hz_reproduction \
  --train dataset2 --epochs 30 --device mps --output outputs/experiment6_reproduction
```

Use `--device cpu` if MPS is unavailable. Fresh training may produce different
checkpoints/results across hardware or package versions.

### Evaluate the saved completed models without retraining

```bash
OMP_NUM_THREADS=1 python learning_pipeline/non_diagnostic_leads.py train \
  --dataset1 data/dataset1_500hz --dataset2 data/dataset2_500hz --train dataset1 \
  --checkpoint outputs/experiment5_500hz/best_model.pt --device mps \
  --output outputs/experiment5_evaluation_repeat
OMP_NUM_THREADS=1 python learning_pipeline/non_diagnostic_leads.py train \
  --dataset1 data/dataset1_500hz --dataset2 data/dataset2_500hz --train dataset2 \
  --checkpoint outputs/experiment6_500hz/best_model.pt --device mps \
  --output outputs/experiment6_evaluation_repeat
```

Although the subcommand is named `train`, supplying `--checkpoint` skips all
optimizer steps. The checkpoint's training export fingerprints, architecture,
dataset identity and lead/label order must match.

## Artifacts and checks

| Location | Contents |
|---|---|
| `data/dataset1_500hz/`, `data/dataset2_500hz/` | Local validated manifests, contracts and native six-lead waveform/label arrays |
| `outputs/experiment5_500hz/best_model.pt` | Preserved Experiment 5 weights |
| `outputs/experiment5_both_tests/` | Both Experiment 5 evaluations, scores, export provenance and leakage checks |
| `outputs/experiment6_500hz/` | Experiment 6 weights, epoch history, configuration, both evaluations and scores |
| `results/experiment5_both_tests/`, `results/experiment6_500hz/` | Small aggregate review bundles without weights or per-record predictions |
| `results/assignment_results.json` | All eight per-label result rows and completion/preservation checks |
| `results/dataset_sources_verification.json` | Exact matches to both pinned cohort builders |

The suite passes **15 tests**. Model shape, BCE outputs, source labeling policies,
Dataset 2 overlap assertion, lead selection, source frequency, patient checks,
cross-dataset alignment and evaluation-only mode are covered. Existing WFDB/pandas
deprecation warnings do not prevent checks from passing. Dependency consistency
checks pass. Saved truths/scores reproduce all final metrics, checkpoints match
minimum validation loss, and the existing Experiment 5 checkpoint is unchanged.

The two final `best_model.pt` checkpoints are included in the repository (about
1 MB each) at the paths listed above. Other generated outputs, raw data, arrays,
per-record predictions and caches remain Git-ignored. A fresh checkout contains
code, aggregate reports and both final checkpoints; ECG data must be supplied locally.
Checkpoints preserve parameters and provenance; they do not contain all state
needed for exact interrupted-training resumption.

## Limitations

- Dataset 1 follows the designated branch's diagnostic-only negative filter,
  permitting abnormal form/rhythm statements; it is not a disease-free cohort.
- ILBBB has only 62 training, seven validation and eight test positives. One
  test miss changes recall by 12.5 percentage points. Read precision, recall and
  confusion matrices together; high negative-class accuracy can be misleading.
- The source fold 10 has previous development exposure. Results are internal
  held-out evaluations, not untouched final external validation. No settings were
  adjusted using these scores.
- Dataset 1 test is contained in Dataset 2 test, so the evaluations are correlated.
  Patient separation does not exclude device/age confounding, transformed copies,
  errors in source IDs, or other acquisition shortcuts.
- No clinical deployment validation, external validation or feature-stability
  analysis is claimed by these experiments.
