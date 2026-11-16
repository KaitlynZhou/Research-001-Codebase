# Experiment 2 — Evaluation Results

**Model evaluated:** Saved 12-lead ResNet checkpoint from epoch 12 (trained on Dataset 2: LBBB vs. all other PTB-XL ECGs).  
**Best observed validation loss:** 0.4142.  
**Primary classification threshold:** 0.50 for both CLBBB and ILBBB.  
**Outputs:** Two separate binary classifications (CLBBB, ILBBB).

> **Training-status caveat:** Training was manually interrupted after 13 completed epochs; the checkpoint from epoch 12 had the lowest validation loss observed during that run. The original 30-epoch training budget was not completed.

## 1. Dataset 2 validation results (fold 9)

Validation population: **2,183 ECG recordings** (54 CLBBB positives, 7 ILBBB positives).

| Metric | CLBBB | ILBBB |
|---|---:|---:|
| Accuracy | 0.9643 | 0.8571 |
| Precision | 0.4032 | 0.0189 |
| Recall | 0.9259 | 0.8571 |
| F1 | 0.5618 | 0.0370 |
| ROC-AUC | 0.9933 | 0.9429 |
| Average precision (PR-AUC summary) | 0.8179 | 0.0564 |
| True negatives (TN) | 2,055 | 1,865 |
| False positives (FP) | 74 | 311 |
| False negatives (FN) | 4 | 1 |
| True positives (TP) | 50 | 6 |

At threshold 0.50, the model detected 50/54 CLBBB and 6/7 ILBBB cases, but produced substantial false positives, especially for ILBBB. ROC-AUC was high for both labels, whereas threshold-based precision and F1 were much lower. The small number of ILBBB-positive validation cases makes its metrics sensitive to individual recordings.

## 2. ILBBB validation threshold sensitivity

Thresholds were explored **only on Dataset 2 validation fold 9**. These results are exploratory and do not change the predefined 0.50 primary threshold used for final testing.

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.0189 | 0.8571 | 0.0370 | 6 | 311 | 1 |
| 0.60 | 0.0214 | 0.8571 | 0.0417 | 6 | 275 | 1 |
| 0.70 | 0.0256 | 0.8571 | 0.0498 | 6 | 228 | 1 |
| 0.80 | 0.0270 | 0.7143 | 0.0521 | 5 | 180 | 2 |
| 0.90 | 0.0339 | 0.5714 | 0.0640 | 4 | 114 | 3 |

Increasing the ILBBB threshold from 0.50 to 0.90 reduced false positives from **311 to 114** but also reduced true positives from **6 to 4**. The highest F1 among these tested thresholds was **0.0640 at 0.90**, still indicating poor precision. No threshold was chosen using the test data.

## 3. Final held-out test results (fold 10)

The **same saved Experiment 2 checkpoint** was evaluated on both test populations using threshold **0.50**, without retraining or test-set threshold tuning.

### 3.1 Dataset 1 test — LBBB vs. clean normal

Population: **974 ECG recordings** (54 CLBBB positives, 8 ILBBB positives).

| Metric | CLBBB | ILBBB |
|---|---:|---:|
| Accuracy | 0.9918 | 0.9148 |
| Precision | 0.8710 | 0.0787 |
| Recall | 1.0000 | 0.8750 |
| F1 | 0.9310 | 0.1443 |
| ROC-AUC | 0.9986 | 0.8662 |
| Average precision | 0.9738 | 0.4072 |
| TN | 912 | 884 |
| FP | 8 | 82 |
| FN | 0 | 1 |
| TP | 54 | 7 |

### 3.2 Dataset 2 test — LBBB vs. all others

Population: **2,198 ECG recordings** (54 CLBBB positives, 8 ILBBB positives).

| Metric | CLBBB | ILBBB |
|---|---:|---:|
| Accuracy | 0.9677 | 0.8562 |
| Precision | 0.4320 | 0.0217 |
| Recall | 1.0000 | 0.8750 |
| F1 | 0.6034 | 0.0424 |
| ROC-AUC | 0.9973 | 0.8613 |
| Average precision | 0.8675 | 0.1537 |
| TN | 2,073 | 1,875 |
| FP | 71 | 315 |
| FN | 0 | 1 |
| TP | 54 | 7 |

### 3.3 Cross-test interpretation

- **CLBBB:** All 54 positives were detected in both test sets (recall 1.00). Precision was higher against clean-normal controls (**0.8710**) than against the broader Dataset 2 negatives (**0.4320**), where the model generated 71 false positives instead of 8.
- **ILBBB:** Seven of eight positives were detected in both test sets (recall **0.8750**), but precision remained low: **0.0787** on Dataset 1 and **0.0217** on Dataset 2. The broader negative population produced **315** false positives versus **82**.
- **Interpretation caution:** Dataset 1's test recordings are a subset of Dataset 2's test recordings. The two reported test evaluations are **not independent cohorts**, and their prevalence and negative-case composition differ. Precision and F1 therefore should not be compared without this qualification.

## 4. Common-test-set comparison with Experiment 1

Both models were reported on the **same Dataset 1 held-out test population (974 recordings)** at threshold 0.50. Experiment 1 figures come from its recorded experiment summary; Experiment 2 figures come from the current notebook outputs.

| Dataset 1 test metric | Experiment 1: trained on Dataset 1 | Experiment 2: trained on Dataset 2 |
|---|---:|---:|
| CLBBB precision | 0.813 | **0.871** |
| CLBBB recall | 0.963 | **1.000** |
| CLBBB F1 | 0.881 | **0.931** |
| CLBBB ROC-AUC | 0.998 | 0.999 (0.9986) |
| ILBBB precision | 0.076 | **0.079** |
| ILBBB recall | 0.875 | 0.875 |
| ILBBB F1 | 0.140 | **0.144** |
| ILBBB ROC-AUC | **0.875** | 0.866 |

The Experiment 2 checkpoint had higher CLBBB F1 on the common Dataset 1 test population. ILBBB F1 was similar and low in both experiments. These are descriptive comparisons of individual training runs, not evidence that the broader training population alone caused the differences. Experiment 2 training was interrupted before its planned maximum, and no repeated-seed uncertainty assessment was performed.

## 5. Evaluation limitations

1. **Rare ILBBB positives:** Only 7 validation and 8 test ILBBB-positive recordings; precision, recall, and threshold analyses are unstable.
2. **Strong class imbalance:** Weighted BCE prioritized positive examples, but the final model still produced many false positives at threshold 0.50. Weighted-loss outputs should not automatically be interpreted as calibrated probabilities.
3. **Incomplete training budget:** The saved epoch-12 checkpoint was best among the 13 completed epochs; later epochs were not evaluated.
4. **Overlapping test populations:** Dataset 1 test is a subset of Dataset 2 test, not an independent external validation set.
5. **No test-set tuning:** All final test metrics use the prespecified 0.50 threshold. Validation threshold sensitivity was reported separately.

## 6. Locally saved evaluation artifacts

```text
results/experiment2/
├── best_model.pt
├── training_history.csv
├── validation_metrics.csv
├── validation_scores.npy
├── ilbbb_threshold_comparison.csv
├── dataset1_test_scores.npy
├── dataset1_test_metrics.csv
├── dataset2_test_scores.npy
├── dataset2_test_metrics.csv
└── both_test_metrics.csv
```

These paths describe the files saved by the notebook cells. Large weights and prediction arrays should remain local rather than being committed to GitHub.
