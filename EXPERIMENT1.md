# Experiment 2 - 12 Leads

## Objective
Establish a baseline for detecting **complete left bundle branch block (CLBBB)** and **incomplete left bundle branch block (ILBBB)** from standard 12-lead ECGs. This baseline supports comparison with Experiment 2.

## Dataset
- **Source:** PTB-XL, 100 Hz ECG recordings (10 seconds each).
- **Dataset 1 cohort:** Include recordings labeled CLBBB or ILBBB, plus clean-normal recordings (NORM without other diagnostic codes). Other recordings are excluded.
- **Inputs:** All 12 standard leads, shape `(12, 1000)` per ECG; saved as `float32` without additional normalization in the preprocessing script.
- **Targets:** Two binary labels, `[CLBBB, ILBBB]`.
- **Split:** Official PTB-XL stratified folds **1–8 train, 9 validation, 10 test**; patient overlap is checked.

## Model
- **Architecture:** 1D ECGResNet, with a 12→32 channel convolutional stem followed by three residual blocks (32→32, 32→64, 64→128); kernel size 7, downsampling in the last two blocks.
- **Output:** Global average pooling and a linear layer producing **two logits**, one per condition; sigmoid probabilities at evaluation.

## Training
| Setting | Value |
|---|---|
| Random seed | 42 |
| Batch size | 32 |
| Optimizer | Adam, learning rate 0.001 (default weight decay 0) |
| Loss | Weighted `BCEWithLogitsLoss`; per-label positive weights = negative/positive counts in training |
| Maximum epochs | 30 |
| Early stopping | Patience 7, using validation loss |
| Selected checkpoint | Lowest validation-loss model |
| Device | Apple MPS when available; otherwise CPU |

## Evaluation
- Evaluate each label separately on validation and held-out Dataset 1 test recordings.
- Report precision, recall, F1, ROC-AUC, and confusion matrices at the **0.50** decision threshold; test evaluation also reports average precision.
- Inspect ILBBB validation thresholds **0.5–0.9** as an exploratory comparison; the standard reported test threshold remains 0.50.

## Key Decisions and Limitations
1. **Restricted cohort:** Comparing LBBB to clean-normal ECGs simplifies the baseline but does not measure discrimination against other abnormal ECGs.
2. **All 12 leads:** Provides a standard full-lead reference before exploring alternative lead configurations.
3. **Weighted loss:** Addresses imbalance separately for CLBBB and ILBBB.
4. **Patient-aware evaluation:** Uses official folds and checks patient overlap to reduce leakage.
5. **Validation-only model selection:** Keeps test fold 10 out of training and checkpoint selection.

**Implementation:** `learning_pipeline/` on branch `feature/ptbxl-data-processing`. Outputs are stored locally under `data/processed/dataset1/` and `results/experiment1/`. This document describes the code configuration, not verified performance results.
