# Experiment 4: Dataset 2, ResNet, conventional diagnostic leads

Trained one ResNet from scratch on Dataset 2 using I, aVL, V1, V2, V5, V6 at native 500 Hz. Separate sigmoid outputs predict CLBBB and ILBBB. Physical WFDB signals are used without normalization, matching the team preprocessing.

Patient folds: 1–8 training, 9 validation, 10 testing. The best checkpoint uses validation loss only; both held-out evaluations use threshold 0.5. Dataset 1 test is a subset of Dataset 2 test, so these evaluations are not independent cohorts.

Best epoch: 11; validation loss: 0.479960.

| Test set | Label | Accuracy | Precision | Recall | F1 | TN | FP | FN | TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dataset1 | CLBBB | 0.990760 | 0.868852 | 0.981481 | 0.921739 | 912 | 8 | 1 | 53 |
| dataset1 | ILBBB | 0.943532 | 0.087719 | 0.625000 | 0.153846 | 914 | 52 | 3 | 5 |
| dataset2 | CLBBB | 0.978162 | 0.530000 | 0.981481 | 0.688312 | 2097 | 47 | 1 | 53 |
| dataset2 | ILBBB | 0.914468 | 0.026316 | 0.625000 | 0.050505 | 2005 | 185 | 3 | 5 |

Confusion matrices use rows = actual [negative, positive], columns = predicted [negative, positive]: [[TN, FP], [FN, TP]].

Dataset 1 uses the team’s diagnostic-only Normal filter plus all CLBBB/ILBBB records. Dataset 2 uses Kevin’s all-record code-presence labels. Each label is evaluated separately against all other test records.

Patient separation, cross-dataset membership, labels, array shapes, and finite values passed the checks in leakage_checks.json. raw_verification.json records SHA-256 verification of the native signals. training_config.json records exact source commits and settings; history.json contains all completed epochs.

Reproduce from the repository root with `.venv/bin/python -u experiment4/run.py` after placing the official PTB-XL 1.0.3 ZIP at `data/raw/ptb-xl-1.0.3.zip`. Install `experiment4/requirements-lock.txt` in a Python 3.13 environment.
