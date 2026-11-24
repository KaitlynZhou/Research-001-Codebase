from pathlib import Path

import numpy as np


LEADS = ("I", "AVL", "V5", "V6")
TARGETS = ("CLBBB", "ILBBB")
TARGET_MODES = ("separate", "combined")
RESULTS_DIR = Path("results/lead_experiments")


def data_dir(lead):
    return Path("data/processed") / f"dataset1_lead_{lead}"


def target_names(target_mode):
    return ("LBBB",) if target_mode == "combined" else TARGETS


def results_root(target_mode):
    return RESULTS_DIR / "combined" if target_mode == "combined" else RESULTS_DIR


def results_dir(lead, target_mode="separate"):
    return results_root(target_mode) / f"lead_{lead}_10sec"


def load_split(lead, split, target_mode):
    directory = data_dir(lead)
    X = np.load(directory / f"X_{split}.npy")
    y = np.load(directory / f"y_{split}.npy")
    if X.ndim != 3 or X.shape[1:] != (1, 1000) or len(X) == 0:
        raise ValueError(f"{lead} {split}: expected nonempty (N, 1, 1000), got {X.shape}")
    if y.shape != (len(X), len(TARGETS)) or not np.isin(y, (0, 1)).all():
        raise ValueError(f"{lead} {split}: expected two binary labels per recording")

    if target_mode == "combined":
        y = np.logical_or(y[:, :1], y[:, 1:2])
    return X, y.astype(np.float32, copy=False)
