# Six diagnostic leads: Dataset 1 baseline

Training and evaluation were performed locally on PTB-XL 1.0.3. Source code and aggregate results are published on `feature/dataset1-diagnostic-leads`.

## Method

Reference: Prisha’s `feature/dataset1-12lead-clean` branch at commit `4d82a777d7b811ea35e57d1abfc9cfb5c8128f6e`. Leads: I, aVL, V1, V2, V5, V6. Separate CLBBB and ILBBB outputs; any recording containing a target code is positive for that output, including recordings with additional diagnoses. CLBBB is the primary outcome; ILBBB is secondary.

Same three-block 1D ResNet and two-output weighted BCE as Prisha; change input channels from 12 to 6. Adam LR 0.001, batch 32, seed 42, maximum 30 epochs, early stopping patience 7, no waveform normalization or augmentation. Training class weights: [17.06775665283203, 123.7258071899414].

Train: Dataset 1 folds 1–8 (7,733 recordings, 428 CLBBB / 62 ILBBB). Validation: Dataset 1 fold 9 (975 recordings, 54 / 7). Test: fold 10 only. Patient disjointness and exact waveform duplicate checks passed.

Completed 10 epochs in 13.6 minutes; selected epoch 3 by minimum validation weighted BCE (0.325712). Parameters: 243,234. Device: cpu.

## Held-out results

One frozen checkpoint, fixed threshold 0.50 for both outcomes. No test-set threshold tuning.

| Test population | Outcome | N | Positives | ROC-AUC | Average precision | Precision | Sensitivity | Specificity | F1 | TN | FP | FN | TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dataset1_test | CLBBB | 974 | 54 | 0.9974 | 0.9506 | 0.8667 | 0.9630 | 0.9913 | 0.9123 | 912 | 8 | 2 | 52 |
| dataset1_test | ILBBB | 974 | 8 | 0.9200 | 0.2136 | 0.0814 | 0.8750 | 0.9182 | 0.1489 | 887 | 79 | 1 | 7 |
| dataset2_test | CLBBB | 2198 | 54 | 0.9867 | 0.6569 | 0.1948 | 0.9630 | 0.8997 | 0.3240 | 1929 | 215 | 2 | 52 |
| dataset2_test | ILBBB | 2198 | 8 | 0.8919 | 0.1300 | 0.0150 | 0.8750 | 0.7900 | 0.0295 | 1730 | 460 | 1 | 7 |

## Interpretation and limitations

Dataset 1 test (974 recordings) is a subset of Dataset 2 test (2,198 recordings). These are overlapping cohorts from the same dataset, not independent external validation. All 54 CLBBB and 8 ILBBB test positives occur in both cohorts; therefore sensitivity is identical. The 1,224 additional test recordings have other diagnoses and no target labels. Their false-positive counts are reported separately in metrics.csv.

ILBBB is exploratory: only 62 training, 7 validation, and 8 test positives. Each missed test positive changes sensitivity by 12.5 percentage points. ROC-AUC alone is insufficient for this rare outcome; inspect average precision, prevalence, and false positives. Zero precision/F1 when there are no predicted positives follows the stated zero_division=0 convention. AUROC, average precision and sensitivity are undefined for the additional-diagnoses subset because it contains no positives.

Sigmoid outputs from class-weighted BCE are model scores, not calibrated clinical probabilities. This diagnostic experiment does not estimate sudden cardiac death risk or establish clinical utility. No interpretability/VAE experiments were run.

The six-lead network has 1,344 fewer parameters than the original model. Training hyperparameters were held fixed to isolate the requested lead change. Device/version differences can prevent bitwise reproduction of Prisha’s run. A single run cannot establish feature stability across seeds.

## Local artifacts

See README.md for execution commands and Explore_baseline.ipynb for local inspection. The repository’s results/ contains configuration, epoch history and aggregate metrics. Checkpoints, record-level predictions, ECG arrays and generated plots remain local. provenance.json records GitHub references. waveform_audit.json records duplicate checks. data_audit.json records cohort counts and metadata checksum. Raw/processed ECG arrays are retained separately in the workspace and omitted from the portable package.
