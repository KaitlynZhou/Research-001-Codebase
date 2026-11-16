# Dataset 1 baseline: six diagnostic leads

This summary describes the **six-diagnostic-lead experiment** on `feature/dataset1-diagnostic-leads`.

The branch was created from Prisha's 12-lead baseline. The previous contents of this file described that inherited experiment. The original report remains available on [Prisha's 12-lead branch](https://github.com/KaitlynZhou/Research-001-Codebase/blob/4d82a777d7b811ea35e57d1abfc9cfb5c8128f6e/results/experiment1/experiment_summary.md).

See the [full diagnostic-lead report](../../diagnostic_leads/REPORT.md), [training/evaluation instructions](../../diagnostic_leads/README.md), and [six-lead metrics CSV](../../diagnostic_leads/results/metrics.csv).

## Method

- Inputs: **I, aVL, V1, V2, V5, V6**, 100 Hz, 10 seconds; shape `(6, 1000)`.
- Model: three-block 1D ResNet, **243,234 parameters**, with independent CLBBB and ILBBB outputs.
- Labels: target-code presence defines each outcome independently; recordings with other diagnoses remain positive for the corresponding outcome.
- Train on Dataset 1 (LBBB plus normal controls): folds 1–8, 7,733 recordings.
- Validate on Dataset 1 fold 9: 975 recordings.
- Follow Prisha's training settings: seed 42, Adam learning rate 0.001, batch size 32, training-derived positive weights, weighted binary cross-entropy, maximum 30 epochs, early-stopping patience 7.
- Training stopped after **10 epochs**; the best checkpoint was **epoch 3**, validation loss **0.325712**.
- Evaluate the same checkpoint on fold 10 of both test populations, using threshold **0.50** for each outcome.

## Dataset 1 test: LBBB versus normal

974 recordings; 54 CLBBB positives and 8 ILBBB positives.

| Outcome | ROC-AUC | Average precision | Precision | Sensitivity | Specificity | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CLBBB | 0.9974 | 0.9506 | 0.8667 | 0.9630 | 0.9913 | 0.9123 | 912 | 8 | 2 | 52 |
| ILBBB | 0.9200 | 0.2136 | 0.0814 | 0.8750 | 0.9182 | 0.1489 | 887 | 79 | 1 | 7 |

## Dataset 2 test: LBBB versus all others

2,198 recordings; 54 CLBBB positives and 8 ILBBB positives. This is evaluation of the **Dataset 1-trained six-lead model**, not a separately Dataset 2-trained model.

| Outcome | ROC-AUC | Average precision | Precision | Sensitivity | Specificity | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CLBBB | 0.9867 | 0.6569 | 0.1948 | 0.9630 | 0.8997 | 0.3240 | 1929 | 215 | 2 | 52 |
| ILBBB | 0.8919 | 0.1300 | 0.0150 | 0.8750 | 0.7900 | 0.0295 | 1730 | 460 | 1 | 7 |

## Interpretation

Six diagnostic leads achieved strong CLBBB discrimination against normal controls. Compared with Prisha's 12-lead run, Dataset 1 CLBBB false positives decreased from 12 to 8, with the same 52/54 detected positives. The 12-lead model had slightly higher ROC-AUC and average precision; this single run does not establish overall superiority.

Precision decreased substantially against the broader Dataset 2 population. ILBBB remained exploratory because there were only 62 training, 7 validation, and 8 test positives, with many false positives.

Dataset 1 test is a subset of Dataset 2 test; these overlapping PTB-XL cohorts are not external validation. Their positive cases are identical, so sensitivity is identical. Both test cohorts are patient-disjoint from this model's training and validation cohorts. No test-based threshold tuning was used.

## File locations

All six-lead source code and result artifacts are under [diagnostic_leads/](../../diagnostic_leads/). Other files still present under `learning_pipeline/` and `results/experiment1/`, including the neighboring metrics CSVs, are inherited **12-lead reference artifacts**. Use `diagnostic_leads/results/metrics.csv` for this experiment's numerical results.
