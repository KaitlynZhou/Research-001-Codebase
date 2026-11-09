"""Reproducible single-lead ECG baseline. Synthetic results are software tests only."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             precision_recall_curve, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, auc)
from torch import nn
from torch.utils.data import Dataset, DataLoader

from ptbxl_data import load_v6_exports


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Unsupported nondeterministic operations should fail instead of silently varying.
    torch.use_deterministic_algorithms(True)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ECGDataset(Dataset):
    """Accept finite V6 arrays shaped (N, 5000) or (N, 1, 5000)."""
    def __init__(self, signals, labels):
        x = np.asarray(signals, dtype=np.float32)
        y = np.asarray(labels)
        if x.ndim == 2:
            x = x[:, None, :]
        if x.ndim != 3 or x.shape[1:] != (1, 5000) or not len(x):
            raise ValueError("Signals must have nonempty shape (N, 5000) or (N, 1, 5000)")
        if y.shape != (len(x),) or not np.isin(y, [0, 1]).all():
            raise ValueError("Labels must have shape (N,) and contain only 0 (Normal) or 1 (LBBB)")
        if not np.isfinite(x).all():
            raise ValueError("Signals contain nonfinite values")
        self.x = torch.from_numpy(x.copy())
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 7, stride, 3, bias=False),
            nn.BatchNorm1d(out_channels), nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 7, padding=3, bias=False),
            nn.BatchNorm1d(out_channels))
        self.skip = (nn.Identity() if in_channels == out_channels and stride == 1 else
                     nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                                   nn.BatchNorm1d(out_channels)))
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.main(x) + self.skip(x))


class ResNet1D(nn.Module):
    """Two logits per recording; class 1 represents LBBB."""
    def __init__(self, base_channels=16):
        super().__init__()
        self.base_channels = base_channels
        self.features = nn.Sequential(
            nn.Conv1d(1, base_channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels), nn.ReLU(),
            ResidualBlock(base_channels, base_channels),
            ResidualBlock(base_channels, base_channels * 2, 2),
            ResidualBlock(base_channels * 2, base_channels * 4, 2),
            nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Linear(base_channels * 4, 2)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (1, 5000):
            raise ValueError("ResNet expects (N, 1, 5000) V6 recordings at 500 Hz")
        return self.classifier(self.features(x).squeeze(-1))


def load_processed_data(path):
    """Load teammate-provided NPZ arrays; fail closed if patient metadata is absent."""
    result = {}
    patient_sets = []
    with np.load(path, allow_pickle=False) as arrays:
        for split in ("train", "val", "test"):
            required = [f"X_{split}", f"y_{split}", f"patient_ids_{split}"]
            if any(key not in arrays for key in required):
                raise ValueError(f"Required arrays: {required}; patient IDs are mandatory")
            dataset = ECGDataset(arrays[required[0]], arrays[required[1]])
            ids = arrays[required[2]]
            if ids.shape != (len(dataset),):
                raise ValueError(f"patient_ids_{split} must have one ID per recording")
            canonical = [str(value).strip() for value in ids.tolist()]
            if any(value.lower() in ("", "nan", "none", "nat") for value in canonical):
                raise ValueError("Missing patient IDs are not allowed")
            patient_sets.append(set(canonical))
            result[split] = dataset
    for i in range(3):
        for j in range(i + 1, 3):
            if patient_sets[i] & patient_sets[j]:
                raise ValueError("Patient overlap between splits: possible data leakage")
    return result


def synthetic_data(seed=42, sizes=(128, 32, 32)):
    """Artificial waveforms, not simulated diagnoses or clinical evidence."""
    rng = np.random.default_rng(seed)
    result = {}
    t = np.arange(5000, dtype=np.float32) / 500
    for split, n in zip(("train", "val", "test"), sizes):
        labels = np.arange(n) % 2
        rng.shuffle(labels)
        signals = rng.normal(0, .2, (n, 5000))
        signals += np.sin(2 * np.pi * (2 + labels[:, None]) * t)
        result[split] = ECGDataset(signals, labels)
    return result


def make_loaders(datasets, batch_size=32, seed=42):
    generator = torch.Generator().manual_seed(seed)
    return {name: DataLoader(data, batch_size=batch_size, shuffle=name == "train",
                             generator=generator if name == "train" else None,
                             num_workers=0) for name, data in datasets.items()}


def training_class_weights(dataset):
    """Balanced inverse-frequency weights computed exclusively from training labels."""
    counts = torch.bincount(dataset.y, minlength=2).float()
    if (counts == 0).any():
        raise ValueError("Training requires both classes for balanced loss")
    return len(dataset) / (2 * counts)


def run_epoch(model, loader, device, optimizer=None, class_weights=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = denominator = correct = count = 0
    true_positive = false_positive = false_negative = 0
    weights = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weights, reduction="sum")
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            batch_denominator = weights[y].sum() if weights is not None else len(y)
            total_loss = criterion(logits, y)
            loss = total_loss / batch_denominator
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += total_loss.item()
            denominator += float(batch_denominator)
            predicted = logits.argmax(1)
            correct += (predicted == y).sum().item()
            true_positive += ((predicted == 1) & (y == 1)).sum().item()
            false_positive += ((predicted == 1) & (y == 0)).sum().item()
            false_negative += ((predicted == 0) & (y == 1)).sum().item()
            count += len(y)
    return {"loss": loss_sum / denominator, "accuracy": correct / count,
            "recall": true_positive / max(true_positive + false_negative, 1),
            "f1": 2 * true_positive / max(2 * true_positive + false_positive + false_negative, 1)}


def train(model, loaders, device, output_dir, epochs=10, learning_rate=1e-3,
          metadata=None, patience=5, min_delta=1e-4, balance_classes=True):
    if patience < 1 or min_delta < 0:
        raise ValueError("patience must be positive and min_delta nonnegative")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    weights = training_class_weights(loaders["train"].dataset) if balance_classes else None
    config = {"max_epochs": epochs, "patience": patience, "min_delta": min_delta,
              "learning_rate": learning_rate, "batch_size": loaders["train"].batch_size,
              "class_weights": weights.tolist() if weights is not None else None,
              "selection_metric": "validation weighted cross entropy" if weights is not None
                                  else "validation cross entropy"}
    (output_dir / "training_config.json").write_text(json.dumps(config, indent=2))
    history = {f"{split}_{metric}": [] for split in ("train", "val")
               for metric in ("loss", "accuracy", "recall", "f1")}
    stopping_reference, stale_epochs, best_epoch = float("inf"), 0, 0
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        train_stats = run_epoch(model, loaders["train"], device, optimizer, weights)
        val_stats = run_epoch(model, loaders["val"], device, class_weights=weights)
        for prefix, stats in (("train", train_stats), ("val", val_stats)):
            for metric, value in stats.items():
                history[f"{prefix}_{metric}"].append(value)
        if val_stats["loss"] < best_loss:
            best_loss = val_stats["loss"]
            best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "base_channels": model.base_channels, "input_samples": 5000,
                        "sampling_rate_hz": 500, "epoch": epoch,
                        "val_loss": best_loss, "training_config": config, "metadata": metadata or {}},
                       output_dir / "best_model.pt")
        print(f"Epoch {epoch}/{epochs}: train loss={train_stats['loss']:.4f}, "
              f"val loss={val_stats['loss']:.4f}, val accuracy={val_stats['accuracy']:.3f}, "
              f"val LBBB recall={val_stats['recall']:.3f}, F1={val_stats['f1']:.3f}", flush=True)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))
        if val_stats["loss"] < stopping_reference - min_delta:
            stopping_reference, stale_epochs = val_stats["loss"], 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch}; best checkpoint: epoch {best_epoch}", flush=True)
            break
    (output_dir / "training_summary.json").write_text(json.dumps({
        "epochs_completed": epoch, "best_epoch": best_epoch, "best_val_loss": best_loss,
        "early_stopped": epoch < epochs, "training_config": config}, indent=2))
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, metric in zip(axes.flat, ("loss", "accuracy", "recall", "f1")):
        for split in ("train", "val"):
            ax.plot(range(1, len(history[f"{split}_{metric}"]) + 1), history[f"{split}_{metric}"], label=split)
        ax.set(xlabel="Epoch", ylabel=metric.title())
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png")
    plt.close(fig)
    return history


def load_checkpoint(path, device=None):
    device = get_device() if device is None else torch.device(device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("input_samples") != 5000 or checkpoint.get("sampling_rate_hz") != 500:
        raise ValueError("Checkpoint must be trained on 500 Hz, 5000-sample V6 recordings")
    model = ResNet1D(checkpoint["base_channels"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    for x, y in loader:
        probabilities.append(model(x.to(device)).softmax(1)[:, 1].cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def evaluate(model, loader, device, output_dir, metadata=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    y, probability = predict(model, loader, device)
    predicted = (probability >= .5).astype(int)
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    metrics = {"accuracy": float(accuracy_score(y, predicted)),
               "precision": float(precision_score(y, predicted, zero_division=0)),
               "recall": float(recall_score(y, predicted, zero_division=0)),
               "f1": float(f1_score(y, predicted, zero_division=0)),
               "confusion_matrix": matrix.tolist(), "auroc": None, "pr_auc": None,
               "average_precision": None, "threshold": .5, "metadata": metadata or {}}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        axes[0].text(j, i, str(value), ha="center", va="center")
    axes[0].set(xlabel="Predicted", ylabel="True", title="Confusion matrix",
                xticks=[0, 1], yticks=[0, 1])
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, probability)
        precision, recall, _ = precision_recall_curve(y, probability)
        metrics.update(auroc=float(roc_auc_score(y, probability)),
                       pr_auc=float(auc(recall, precision)),
                       average_precision=float(average_precision_score(y, probability)))
        axes[1].plot(fpr, tpr)
        axes[2].plot(recall, precision)
    else:
        for ax in axes[1:]:
            ax.text(.1, .5, "Undefined: only one class")
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    axes[2].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    fig.suptitle("SYNTHETIC — software test only" if (metadata or {}).get("synthetic")
                 else "Held-out ECG evaluation")
    fig.tight_layout()
    fig.savefig(output_dir / "evaluation.png")
    plt.close(fig)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    np.savez(output_dir / "predictions.npz", labels=y, probabilities=probability)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data", type=Path, help="Prisha's V6 export directory or legacy NPZ")
    source.add_argument("--synthetic", action="store_true", help="Explicit software-test mode")
    parser.add_argument("--raw-root", type=Path, help="Verify every exported row against local records500 V6")
    parser.add_argument("--output", type=Path, help="Defaults to outputs/real_500hz or outputs/synthetic_500hz")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--unweighted", action="store_true", help="Disable training-derived class weights")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--checkpoint", type=Path, help="Evaluate a saved model without training")
    args = parser.parse_args()
    if not args.data and not args.synthetic:
        args.data = Path("data/processed/v6_500hz")
    if args.raw_root and (args.synthetic or not args.data.is_dir()):
        parser.error("--raw-root requires a V6 export directory")
    if args.output is None:
        args.output = Path("outputs/synthetic_500hz" if args.synthetic else "outputs/real_500hz")
    set_seed(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    provenance = {}
    if args.synthetic:
        datasets = synthetic_data(args.seed)
    elif args.data.is_dir():
        arrays, provenance = load_v6_exports(args.data, args.raw_root)
        datasets = {split: ECGDataset(x, y) for split, (x, y) in arrays.items()}
    elif args.data.is_file():
        datasets = load_processed_data(args.data)
    else:
        parser.error(f"Real data unavailable at {args.data}. Supply Prisha's export directory "
                     "with --data, or use --synthetic for a software test. No download attempted.")
    loaders = make_loaders(datasets, args.batch_size, args.seed)
    metadata = {"synthetic": args.synthetic, "seed": args.seed,
                "data_path": str(args.data) if args.data else None,
                "lead": "V6", "sampling_rate_hz": 500, "torch_version": str(torch.__version__),
                "device": str(device), "data_provenance": provenance}
    (args.output).mkdir(parents=True, exist_ok=True)
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Device: {device}; synthetic: {metadata['synthetic']}")
    if args.checkpoint:
        model, checkpoint = load_checkpoint(args.checkpoint, device)
        if checkpoint["metadata"].get("synthetic") != metadata["synthetic"]:
            raise ValueError("Checkpoint and evaluation data must have the same synthetic/real status")
        saved_hashes = checkpoint["metadata"].get("data_provenance", {}).get("file_sha256")
        if saved_hashes is not None and saved_hashes != provenance.get("file_sha256"):
            raise ValueError("Evaluation exports differ from the checkpoint's training data exports")
    else:
        model = ResNet1D()
        train(model, loaders, device, args.output, args.epochs, args.learning_rate, metadata,
              patience=args.patience, min_delta=args.min_delta, balance_classes=not args.unweighted)
        model, _ = load_checkpoint(args.output / "best_model.pt", device)
    # Test data is accessed only after validation-based model selection is complete.
    print(json.dumps(evaluate(model, loaders["test"], device, args.output, metadata), indent=2))


if __name__ == "__main__":
    main()
