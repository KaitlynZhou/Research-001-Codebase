# Experiment 2 - 12 Leads

## Objective
Train a **12-lead ECG ResNet** to predict complete and incomplete left bundle branch block (**CLBBB**, **ILBBB**) using all PTB-XL recordings. Compare against Experiment 1, which trains only on LBBB and clean-normal ECGs, while preserving its model and training setup.

## Dataset
- **Source:** PTB-XL, all recordings (Dataset 2), including other ECG abnormalities and normal ECGs.
- **Input:** Standard 12-lead, 10-second ECGs sampled at 100 Hz; `(12, 1000)` float32 arrays per recording.
- **Targets:** Two separate binary labels: `[CLBBB, ILBBB]` from `scp_codes`.
- **Splits:** PTB-XL folds **1–8 training, 9 validation, 10 test**; patient consistency and lack of split overlap are checked.
- **Preprocessing:** Verify sample rate, dimensions, lead order, finite values and label alignment; save waveforms in disk-backed NumPy arrays. No extra waveform normalization is applied in the preprocessing code.

## Model
- **Architecture:** Same 1D ECGResNet as Experiment 1: 12→32-channel convolutional stem, three residual blocks (32→32, 32→64, 64→128), global average pooling and a 2-logit classifier.
- **Outputs:** Independent sigmoid probabilities for CLBBB and ILBBB.
- **Initialization:** Train a new model; **do not reuse** Experiment 1 weights.

## Training
| Setting | Configuration |
|---|---|
| Seed | 42 |
| Batch size | 32 |
| Optimizer | Adam, learning rate 0.001; no weight decay |
| Loss | `BCEWithLogitsLoss` with training-set per-label positive weights (negative/positive counts) |
| Max epochs | 30 |
| Early stopping | Patience 7 based on validation loss |
| Checkpoint | Lowest validation-loss model, `results/experiment2/best_model.pt` |
| Device | MPS if available, otherwise CPU |

## Evaluation
- On the validation fold, report per-label precision, recall, F1, ROC-AUC, average precision and confusion matrices at threshold **0.50**.
- Explore ILBBB thresholds **0.5–0.9** on validation; retain **0.50** for reported test comparisons.
- Evaluate the **same Experiment 2 checkpoint** on two held-out populations: **Dataset 2 test** (all fold-10 ECGs) and **Dataset 1 test** (the restricted LBBB/clean-normal fold-10 subset). Report per-label metrics and save prediction scores.

## Key Decisions
1. **Only the training cohort changes** relative to Experiment 1; architecture, hyperparameters, folds and primary evaluation threshold remain fixed.
2. **All 12 leads** remain in use; this experiment tests broader training-data coverage, not reduced-lead performance.
3. **Class weights are recalculated** using Dataset 2's training-label frequencies.
4. **No test-based selection:** The fold-10 test data do not determine the model checkpoint or operating threshold.
5. **Overlapping test populations:** Dataset 1 test is a subset of Dataset 2 test; comparisons are not between independent cohorts.

**Code:** `experiment2/01_build_dataset2.py` through `07_evaluate_test.py` on branch `kevin/experiment2-dataset2-all12`. **Outputs:** `data/processed/dataset2/` and `results/experiment2/`. This describes the implemented setup, not measured experimental performance.
