import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from experiment_config import LEADS, TARGETS, data_dir

RAW_DIR = Path("data/raw/ptb-xl")
METADATA_DIR = Path("data/processed/dataset1")
EXPECTED_LEADS = [
    "I", "II", "III", "AVR", "AVL", "AVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]


def prepare_split(split_name, lead_index, output_dir):
    metadata = pd.read_csv(METADATA_DIR / f"{split_name}.csv")

    signals = []
    labels = []

    for _, row in metadata.iterrows():
        record_path = RAW_DIR / row["filename_lr"]

        signal, fields = wfdb.rdsamp(str(record_path))

        # Verify sampling frequency, dimensions, and lead order
        if fields["fs"] != 100:
            raise ValueError(f"Unexpected sampling frequency: {record_path}")

        if signal.shape != (1000, 12):
            raise ValueError(f"Unexpected shape: {record_path}")

        if fields["sig_name"] != EXPECTED_LEADS:
            raise ValueError(f"Unexpected lead order: {record_path}")

        if not np.isfinite(signal).all():
            raise ValueError(f"Non-finite values: {record_path}")

        signal = signal[:, lead_index:lead_index + 1].T.astype(np.float32)

        signals.append(signal)
        labels.append([row[target] for target in TARGETS])

    X = np.stack(signals)
    y = np.asarray(labels, dtype=np.float32)

    np.save(output_dir / f"X_{split_name}.npy", X)
    np.save(output_dir / f"y_{split_name}.npy", y)

    print(f"{split_name}: X={X.shape}, y={y.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract full 10-second single-lead waveforms.")
    parser.add_argument("--lead", required=True, type=str.upper, choices=LEADS)
    lead = parser.parse_args().lead

    lead_index = EXPECTED_LEADS.index(lead)
    output_dir = data_dir(lead)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Lead {lead} (zero-based index {lead_index})")
    for split_name in ("train", "val", "test"):
        prepare_split(split_name, lead_index, output_dir)
