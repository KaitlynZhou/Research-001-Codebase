import numpy as np
import torch
import importlib.util
from pathlib import Path
import csv
from pathlib import Path

MODEL_FILE = Path(__file__).resolve().parent / "07_resnet_model.py"

spec = importlib.util.spec_from_file_location("ecg_resnet_model", MODEL_FILE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

ECGResNet = module.ECGResNet
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "results" / "experiment1" / "best_model.pt"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True
)

model = ECGResNet().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Loaded checkpoint from epoch:", checkpoint["epoch"])
print("Validation loss:", checkpoint["val_loss"])
print("Device:", device)
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "dataset1"

X_val = np.load(DATA_DIR / "X_val.npy")
y_val = np.load(DATA_DIR / "y_val.npy")

print("Validation ECG shape:", X_val.shape)
print("Validation labels shape:", y_val.shape)
X_tensor = torch.from_numpy(X_val).float().to(device)

with torch.no_grad():
    logits = model(X_tensor)
    probabilities = torch.sigmoid(logits)

y_scores = probabilities.cpu().numpy()

print("Prediction shape:", y_scores.shape)
print("First 5 predictions:")
print(y_scores[:5])
threshold = 0.5
y_pred = (y_scores >= threshold).astype(int)

label_names = ["CLBBB", "ILBBB"]

for i, label in enumerate(label_names):
    true = y_val[:, i]
    pred = y_pred[:, i]
    scores = y_scores[:, i]

    print(f"\n--- {label} ---")
    print("Actual positives:", int(true.sum()))
    print("Predicted positives:", int(pred.sum()))

    print("Precision:", precision_score(true, pred, zero_division=0))
    print("Recall:", recall_score(true, pred, zero_division=0))
    print("F1:", f1_score(true, pred, zero_division=0))
    print("ROC-AUC:", roc_auc_score(true, scores))
    print("Confusion matrix:")
    print(confusion_matrix(true, pred, labels=[0, 1]))
print("\n--- ILBBB Threshold Comparison ---")

true = y_val[:, 1]
scores = y_scores[:, 1]

for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
    pred = (scores >= threshold).astype(int)

    precision = precision_score(true, pred, zero_division=0)
    recall = recall_score(true, pred, zero_division=0)
    f1 = f1_score(true, pred, zero_division=0)

    print(
        f"Threshold: {threshold:.1f} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1:.3f} | "
        f"Predicted positives: {pred.sum()}"
    )
ilbbb_true = y_val[:, 1]
ilbbb_scores = y_scores[:, 1]

ap = average_precision_score(ilbbb_true, ilbbb_scores)

print("\nILBBB Average Precision:", ap)
print("ILBBB Positive Prevalence:", ilbbb_true.mean())

output_dir = Path("results/experiment1")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "validation_metrics.csv"

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "label", "threshold", "precision",
        "recall", "f1", "roc_auc"
    ])

    for i, label in enumerate(label_names):
        true = y_val[:, i]
        scores = y_scores[:, i]
        pred = (scores >= 0.5).astype(int)

        writer.writerow([
            label,
            0.5,
            precision_score(true, pred, zero_division=0),
            recall_score(true, pred, zero_division=0),
            f1_score(true, pred, zero_division=0),
            roc_auc_score(true, scores)
        ])

print("Saved validation metrics to:", output_file)

cm_file = output_dir / "validation_confusion_matrices.csv"

with open(cm_file, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "label", "TN", "FP", "FN", "TP"
    ])

    for i, label in enumerate(label_names):
        true = y_val[:, i]
        pred = (y_scores[:, i] >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            true, pred, labels=[0, 1]
        ).ravel()

        writer.writerow([label, tn, fp, fn, tp])

print("Saved confusion matrices to:", cm_file)

threshold_file = output_dir / "ilbbb_threshold_comparison.csv"

with open(threshold_file, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "threshold", "precision", "recall",
        "f1", "true_positives", "false_positives"
    ])

    true = y_val[:, 1]
    scores = y_scores[:, 1]

    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        pred = (scores >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            true, pred, labels=[0, 1]
        ).ravel()

        writer.writerow([
            threshold,
            precision_score(true, pred, zero_division=0),
            recall_score(true, pred, zero_division=0),
            f1_score(true, pred, zero_division=0),
            int(tp),
            int(fp)
        ])

print("Saved threshold comparison to:", threshold_file)