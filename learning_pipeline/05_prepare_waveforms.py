
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

RAW_DIR = Path("data/raw/ptb-xl")
OUTPUT_DIR = Path("data/processed/dataset1")

EXPECTED_LEADS = [
    "I", "II", "III", "AVR", "AVL", "AVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]


def prepare_split(split_name):
    metadata = pd.read_csv(OUTPUT_DIR / f"{split_name}.csv")

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

        
        signal = signal.T.astype(np.float32)

        signals.append(signal)
        labels.append([row["CLBBB"], row["ILBBB"]])

    X = np.stack(signals)
    y = np.asarray(labels, dtype=np.float32)

    np.save(OUTPUT_DIR / f"X_{split_name}.npy", X)
    np.save(OUTPUT_DIR / f"y_{split_name}.npy", y)

    print(f"{split_name}: X={X.shape}, y={y.shape}")


for split_name in ["train", "val", "test"]:
    prepare_split(split_name)
