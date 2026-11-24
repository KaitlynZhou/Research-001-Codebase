import argparse
import csv
from pathlib import Path
import importlib.util
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiment_config import LEADS, TARGET_MODES, load_split, results_dir, target_names


# --------------------------------------------------
# 1. Reproducibility and configuration
# --------------------------------------------------

SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 30
LEARNING_RATE = 0.001
PATIENCE = 7

parser = argparse.ArgumentParser(description="Train a fresh single-lead ResNet.")
parser.add_argument("--lead", required=True, type=str.upper, choices=LEADS)
parser.add_argument("--target-mode", choices=TARGET_MODES, default="separate")
parser.add_argument("--check-only", action="store_true", help="Check inputs and model without training.")
args = parser.parse_args()
lead = args.lead
targets = target_names(args.target_mode)

OUTPUT_DIR = results_dir(lead, args.target_mode)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print("Device:", device)


# --------------------------------------------------
# 2. Import the ResNet architecture
# --------------------------------------------------

model_path = Path(__file__).with_name("02_resnet_model.py")

spec = importlib.util.spec_from_file_location(
    "resnet_model", model_path
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

model = module.ECGResNet(num_outputs=len(targets)).to(device)


# --------------------------------------------------
# 3. Load training and validation arrays
# --------------------------------------------------

X_train, y_train = load_split(lead, "train", args.target_mode)
X_val, y_val = load_split(lead, "val", args.target_mode)

for split, X, y in (("train", X_train, y_train), ("val", X_val, y_val)):
    counts = dict(zip(targets, y.sum(axis=0).astype(int).tolist()))
    print(f"{lead} {split}: X={X.shape}, y={y.shape}, positives={counts}")

model.eval()
with torch.no_grad():
    logits = model(torch.from_numpy(X_train[:1]).to(device))
if logits.shape != y_train[:1].shape:
    raise ValueError(f"Expected logits shaped {y_train[:1].shape}, got {logits.shape}")
print("Model output shape:", tuple(logits.shape))

train_dataset = TensorDataset(
    torch.from_numpy(X_train),
    torch.from_numpy(y_train)
)

val_dataset = TensorDataset(
    torch.from_numpy(X_val),
    torch.from_numpy(y_val)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --------------------------------------------------
# 4. Handle class imbalance
# --------------------------------------------------

positive_counts = y_train.sum(axis=0)
negative_counts = len(y_train) - positive_counts
if np.any(positive_counts == 0) or np.any(negative_counts == 0):
    raise ValueError("Each training target must have positive and negative examples.")

pos_weight = torch.tensor(
    negative_counts / positive_counts,
    dtype=torch.float32,
    device=device
)

print("Positive class weights:", pos_weight.cpu().numpy())

if args.check_only:
    raise SystemExit(0)

OUTPUT_DIR.mkdir(parents=True, exist_ok=args.target_mode == "separate")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# 5. Training and validation functions
# --------------------------------------------------

def run_epoch(loader, training):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(X)
            loss = criterion(logits, y)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * len(X)
        total_samples += len(X)

    return total_loss / total_samples


# --------------------------------------------------
# 6. Train and save the best checkpoint
# --------------------------------------------------

best_val_loss = float("inf")
epochs_without_improvement = 0

with (OUTPUT_DIR / "training_log.csv").open("w", newline="") as log_file:
    writer = csv.writer(log_file)
    writer.writerow(["epoch", "train_loss", "val_loss", "saved_checkpoint"])

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = run_epoch(train_loader, training=True)
        val_loss = run_epoch(val_loader, training=False)

        print(
            f"Epoch {epoch:02d} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f}"
        )

        improved = val_loss < best_val_loss
        writer.writerow([epoch, train_loss, val_loss, improved])
        log_file.flush()

        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "pos_weight": pos_weight.cpu(),
                    "seed": SEED,
                    "lead": lead,
                    "target_mode": args.target_mode
                },
                OUTPUT_DIR / "best_model.pt"
            )

            print("  Saved new best model")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping triggered")
            break

print("Training complete")
print("Best validation loss:", best_val_loss)
