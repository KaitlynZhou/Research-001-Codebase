import importlib.util
from pathlib import Path
import csv

import torch
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


spec = importlib.util.spec_from_file_location(
    "resnet_model",
    "learning_pipeline/07_resnet_model.py"
)

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

ECGResNet = module.ECGResNet


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


model = ECGResNet().to(device)


checkpoint = torch.load(
    "results/experiment1/best_model.pt",
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded successfully!")
print("Best epoch:", checkpoint["epoch"])
print("Device:", device)

# Load Dataset 1 test arrays
data_dir = Path("data/processed/dataset1")

X_test = np.load(data_dir / "X_test.npy")
y_test = np.load(data_dir / "y_test.npy")

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

print("CLBBB positive cases:", int(y_test[:, 0].sum()))
print("ILBBB positive cases:", int(y_test[:, 1].sum()))


X_test_tensor = torch.tensor(
    X_test, dtype=torch.float32
)


all_scores = []

with torch.no_grad():
    for start in range(0, len(X_test_tensor), 32):
        batch = X_test_tensor[start:start + 32].to(device)

        logits = model(batch)
        probabilities = torch.sigmoid(logits)

        all_scores.append(probabilities.cpu().numpy())


y_scores = np.concatenate(all_scores, axis=0)

print("Prediction scores shape:", y_scores.shape)
print("First 5 prediction scores:")
print(y_scores[:5])


output_dir = Path("results/experiment1")
output_dir.mkdir(parents=True, exist_ok=True)

np.save(
    output_dir / "dataset1_test_scores.npy",
    y_scores
)

print("Saved test prediction scores!")



threshold = 0.5
label_names = ["CLBBB", "ILBBB"]

for i, label in enumerate(label_names):
    true = y_test[:, i]
    scores = y_scores[:, i]
    pred = (scores >= threshold).astype(int)

    print(f"\n--- {label} TEST RESULTS ---")
    print("Actual positives:", int(true.sum()))
    print("Predicted positives:", int(pred.sum()))

    print("Precision:", precision_score(true, pred, zero_division=0))
    print("Recall:", recall_score(true, pred, zero_division=0))
    print("F1:", f1_score(true, pred, zero_division=0))
    print("ROC-AUC:", roc_auc_score(true, scores))
    print("Average Precision:", average_precision_score(true, scores))

    print("Confusion matrix:")
    print(confusion_matrix(true, pred, labels=[0, 1]))
metrics_file = output_dir / "dataset1_test_metrics.csv"

with open(metrics_file, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "label", "threshold", "precision",
        "recall", "f1", "roc_auc",
        "average_precision", "TN", "FP", "FN", "TP"
    ])

    for i, label in enumerate(label_names):
        true = y_test[:, i]
        scores = y_scores[:, i]
        pred = (scores >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            true, pred, labels=[0, 1]
        ).ravel()

        writer.writerow([
            label,
            0.5,
            precision_score(true, pred, zero_division=0),
            recall_score(true, pred, zero_division=0),
            f1_score(true, pred, zero_division=0),
            roc_auc_score(true, scores),
            average_precision_score(true, scores),
            int(tn), int(fp), int(fn), int(tp)
        ])

print("Saved Dataset 1 test metrics to:", metrics_file)
