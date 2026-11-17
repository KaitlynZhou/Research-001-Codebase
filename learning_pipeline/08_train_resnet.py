
from pathlib import Path
import importlib.util
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------
# 1. Reproducibility and configuration
# --------------------------------------------------

SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 30
LEARNING_RATE = 0.001
PATIENCE = 7

DATA_DIR = Path("data/processed/dataset1")
OUTPUT_DIR = Path("results/experiment1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

model_path = Path("learning_pipeline/07_resnet_model.py")

spec = importlib.util.spec_from_file_location(
    "resnet_model", model_path
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

model = module.ECGResNet().to(device)


# --------------------------------------------------
# 3. Load training and validation arrays
# --------------------------------------------------

X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")

X_val = np.load(DATA_DIR / "X_val.npy")
y_val = np.load(DATA_DIR / "y_val.npy")

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

pos_weight = torch.tensor(
    negative_counts / positive_counts,
    dtype=torch.float32,
    device=device
)

print("Positive class weights:", pos_weight.cpu().numpy())

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

for epoch in range(1, MAX_EPOCHS + 1):
    train_loss = run_epoch(train_loader, training=True)
    val_loss = run_epoch(val_loader, training=False)

    print(
        f"Epoch {epoch:02d} | "
        f"Train loss: {train_loss:.4f} | "
        f"Val loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "pos_weight": pos_weight.cpu(),
                "seed": SEED
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
