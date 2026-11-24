import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from experiment_config import (
    LEADS, TARGET_MODES, load_split, results_dir, results_root, target_names
)


BATCH_SIZE = 32
METRIC_COLUMNS = (
    "lead", "prediction_target", "auroc", "average_precision", "positive_test_count"
)


def evaluate(lead, target_mode, model_class, device):
    output_dir = results_dir(lead, target_mode)
    targets = target_names(target_mode)
    checkpoint = torch.load(
        output_dir / "best_model.pt", map_location="cpu", weights_only=True
    )
    if checkpoint.get("lead") != lead:
        raise ValueError(f"Expected a checkpoint trained for lead {lead}: {output_dir}")
    if checkpoint.get("target_mode", "separate") != target_mode:
        raise ValueError(f"Expected a {target_mode} checkpoint: {output_dir}")

    model = model_class(num_outputs=len(targets)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Lead {lead} | Best validation epoch: {checkpoint['epoch']}")

    X_test, y_test = load_split(lead, "test", target_mode)
    X_test = torch.from_numpy(X_test)
    all_scores = []
    with torch.no_grad():
        for batch in X_test.split(BATCH_SIZE):
            probabilities = torch.sigmoid(model(batch.to(device)))
            all_scores.append(probabilities.cpu().numpy())

    y_scores = np.concatenate(all_scores, axis=0)
    rows = []
    for i, target in enumerate(targets):
        true = y_test[:, i]
        scores = y_scores[:, i]
        rows.append((
            lead, target, roc_auc_score(true, scores),
            average_precision_score(true, scores), int(true.sum())
        ))

    np.save(output_dir / "test_scores.npy", y_scores)
    write_metrics(output_dir / "test_metrics.csv", rows)
    return rows


def write_metrics(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(METRIC_COLUMNS)
        writer.writerows(rows)
    print(f"Saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate validation-selected single-lead models.")
    parser.add_argument("--lead", required=True, type=str.upper, choices=(*LEADS, "ALL"))
    parser.add_argument("--target-mode", choices=TARGET_MODES, default="separate")
    args = parser.parse_args()
    lead = args.lead
    target_mode = args.target_mode

    model_path = Path(__file__).with_name("02_resnet_model.py")
    spec = importlib.util.spec_from_file_location("resnet_model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)

    leads = LEADS if lead == "ALL" else (lead,)
    consolidated_path = results_root(target_mode) / "test_metrics.csv"
    output_paths = [consolidated_path] if lead == "ALL" else []
    for selected_lead in leads:
        output_dir = results_dir(selected_lead, target_mode)
        checkpoint_path = output_dir / "best_model.pt"
        if not checkpoint_path.is_file():
            parser.error(f"Train lead {selected_lead} first. Missing {checkpoint_path}")
        output_paths.extend((output_dir / "test_scores.npy", output_dir / "test_metrics.csv"))

    if target_mode == "combined":
        for path in output_paths:
            if path.exists():
                parser.error(f"Combined evaluation output already exists: {path}")

    rows = []
    for selected_lead in leads:
        rows.extend(evaluate(selected_lead, target_mode, module.ECGResNet, device))

    if lead == "ALL":
        write_metrics(consolidated_path, rows)
