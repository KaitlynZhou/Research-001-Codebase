
# Experiment 1: 12-Lead ResNet

## Objective
Classify CLBBB and ILBBB separately using 12-lead ECG recordings from PTB-XL.

## Dataset
- Dataset 1: LBBB vs. Normal
- Training recordings: 7,733
- Validation recordings: 975
- Test recordings: 974 (not yet evaluated)
- Input shape: (12, 1000)
- Patient-disjoint train/validation/test splits

## Model
- Architecture: 1D ResNet
- Trainable parameters: 244,578
- Output: 2 independent logits (CLBBB, ILBBB)

## Training
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 32
- Loss: BCEWithLogitsLoss with positive-class weighting
- Maximum epochs: 30
- Early stopping patience: 7
- Best checkpoint: Epoch 7
- Best validation loss: 0.2651
## Key decisions:

-Used weighted binary cross-entropy to address severe class imbalance.

-Used early stopping based on validation loss to limit overfitting.

-Saved the checkpoint with the lowest validation loss rather than the final epoch.

-Used the same saved checkpoint for validation and test evaluation.
___________________________________________________
## Validation Results (Threshold = 0.50)

| Metric | CLBBB | ILBBB |
|---|---:|---:|
| Precision | 0.909 | 0.061 |
| Recall | 0.926 | 0.857 |
| F1 | 0.917 | 0.113 |
| ROC-AUC | 0.997 | 0.977 |
## Confusion matrices: 
|Condition | TN | FP | FN | TP|
|----------|----|----|----|---|
|CLBBB     | 916 | 5 | 4| | 50|
|ILBBB     | 875 | 93| 1  | 6 | 
-CLBBB demonstrated strong validation performance. ILBBB achieved high recall and ROC-AUC but very low precision due to numerous false positives. 
Thus, an experiment was done with different classificationt thresholds on the validation set only, without retraining the model. 
 
| Threshold | Precision | Recall | F1 Score | False Positives |
|-----------|-----------|--------|----------|-----------------|
| 0.50 | 0.061 | 0.857 | 0.113 | 93 |
| 0.60 | 0.071 | 0.857 | 0.130 | 79 |
| 0.70 | 0.083 | 0.857 | 0.152 | 66 |
| 0.80 | 0.118 | 0.857 | 0.207 | 45 |
| 0.90 | 0.188 | 0.857 | 0.308 | 26 |
## Observations:
-Increasing the threshold reduced false positives from 93 to 26.
-Recall remained constant at 0.857 across the tested thresholds.
-Precision and F1 improved as the threshold increased.
-The 0.90 threshold performed best among those tested, but precision remained relatively low.
*Decision for final threshold choice is still inconclusive and will be discussed with team members later

--------------------------------

## 7. Held-Out Test Results — Dataset 1

Classification threshold: **0.50 for both conditions**

| Metric | CLBBB | ILBBB |
|--------|-------|-------|
| Precision | 0.813 | 0.076 |
| Recall | 0.963 | 0.875 |
| F1 | 0.881 | 0.140 |
| ROC-AUC | 0.998 | 0.875 |
| Average Precision | 0.963 | 0.280 |

**Confusion Matrices:**

| Condition | TN | FP | FN | TP |
|-----------|----|----|----|----|
| CLBBB | 908 | 12 | 2 | 52 |
| ILBBB | 881 | 85 | 1 | 7 |

## Observations:

-CLBBB maintained strong performance on unseen recordings, correctly identifying 52 of 54 positive cases.

-ILBBB continued to demonstrate high recall but low precision. The model detected 7 of 8 positive cases but generated 85 false positives.

-ILBBB ROC-AUC decreased from 0.977 on validation to 0.875 on the test set, indicating less consistent ranking performance across splits. However, the small number of positive ILBBB cases limits confidence in this comparison.

----------------------------------------------------
Limitations: 
-Since we are testing CLBBB and ILBBB seperating, this created a major limitation as ILBBB has only 77 total recordings, and only 62 of those are being used for positive training recordings. Only 7 ILBBB positve recordings are used for validation, and 8 for testing. This limited the amount of positive data available for learning. 
-ILBBB recieved a positive class weight of approximately 123.73, which may have encouraged the model to prioritize recall over precision. 
-Though increasing threshold improved validation performance, optimal threshold has yet to be established because validation improvement may not generalize to unseen recordings. 
