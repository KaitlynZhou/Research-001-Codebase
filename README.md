# Research-001-Codebase
## V6 ResNet baseline

This pipeline supports the project **Evaluating the Stability of Neural
Network-Identified ECG Features**. It classifies Normal (0) versus LBBB (1)
using a single V6 lead at 100 Hz, 1000 samples per recording. Synthetic signals
only exercise software; their metrics are not clinical results or evidence of
feature stability. Feature attribution and stability experiments are future work.

### Install and test

Use Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

### Train and evaluate

```bash
# Small synthetic pipeline test; no downloads required.
python resnet_model.py --epochs 2 --output outputs/synthetic
# Real processed data, once available.
python resnet_model.py --data data/ptbxl_v6.npz --epochs 30 --output outputs/real
# Evaluate a previously selected checkpoint without retraining.
python resnet_model.py --data data/ptbxl_v6.npz \
  --checkpoint outputs/real/best_model.pt --output outputs/real_evaluation
```

CUDA, Apple Silicon MPS, and CPU are selected in that order. Override with
`--device cpu` (or `cuda`/`mps`). Seeds default to 42; set `--seed` explicitly for
repeated experiments. Deterministic operations are requested, but identical results
across different devices or PyTorch versions are not guaranteed. If a backend
reports an unsupported deterministic operation, use `--device cpu`.

The model uses three residual blocks, batch normalization, ReLU, projected skip
connections when dimensions change, and global average pooling. Two output logits
feed CrossEntropyLoss; Adam optimizes the weights. Class 1 softmax probability
is the LBBB score. Predictions use a fixed 0.5 threshold.

Each run saves `best_model.pt` (lowest validation loss, including optimizer state
and configuration), `history.json`, `training_curves.png`, `metrics.json`,
`evaluation.png`, and `predictions.npz`. Evaluation reloads the best checkpoint
before accessing the test loader. Metrics include accuracy, precision, recall,
F1, confusion matrix (rows=true, columns=predicted; order 0,1), AUROC,
trapezoidal PR-AUC, and average precision (reported separately because it differs
from trapezoidal PR-AUC). AUROC and PR-AUC are null for single-class test sets;
precision/recall/F1 use zero for undefined divisions.

### Teammate data interface

Provide a NumPy `.npz` archive with these exact keys:

| Split | Signal array | Label array | Patient array |
| --- | --- | --- | --- |
| Training | `X_train` | `y_train` | `patient_ids_train` |
| Validation | `X_val` | `y_val` | `patient_ids_val` |
| Testing | `X_test` | `y_test` | `patient_ids_test` |

Signals must be finite float arrays shaped `(N, 1000)` or `(N, 1, 1000)`.
Labels must be shaped `(N,)`, with Normal=0 and LBBB=1. Patient arrays must have
one nonmissing, consistently encoded identifier per recording. Use numeric or
Unicode arrays, not object arrays; loading disables pickle. Repeated patients
within one split are allowed, but any patient overlap across splits is rejected.
Missing patient metadata is rejected rather than assuming the splits are safe.

Example archive creation after preprocessing and patient-level splitting:

```python
np.savez("data/ptbxl_v6.npz",
         X_train=X_train, y_train=y_train, patient_ids_train=patient_ids_train,
         X_val=X_val, y_val=y_val, patient_ids_val=patient_ids_val,
         X_test=X_test, y_test=y_test, patient_ids_test=patient_ids_test)
```

The preprocessing owner must select V6, verify 100 Hz sampling and 10-second
length, define Normal/LBBB inclusion and ambiguous-label handling, and split by
patient before exporting. Prefer the project's agreed patient-separated PTB-XL
fold protocol. Fit any dataset-level normalization or preprocessing parameters
on training data only and apply them unchanged to validation/test. This loader
does not infer lead identity, sampling rate, diagnosis validity, or preprocessing
provenance from the arrays, and does not normalize them automatically. Supply
these details with the data handoff. Never tune architecture, epochs, threshold,
or preprocessing using test results. Inspect class balance and ensure both
classes are represented in each split before real experiments.

Smoke tests cover both accepted input shapes, forward/backward passes, weight
updates, training/validation, predictions, checkpoint restoration, artifacts,
reproducibility, invalid arrays, single-class evaluation, and patient leakage
rejection. No PTB-XL data is downloaded by this repository.
