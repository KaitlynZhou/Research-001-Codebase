
import pandas as pd
import wfdb
from pathlib import Path


df = pd.read_csv("data/processed/dataset1/metadata.csv")


record = df.iloc[0]


record_path = Path("data/raw/ptb-xl") / record["filename_lr"]


signal, fields = wfdb.rdsamp(str(record_path))

print("ECG ID:", record["ecg_id"])
print("Signal shape:", signal.shape)
print("Sampling frequency:", fields["fs"])
print("Lead names:", fields["sig_name"])

import numpy as np

print("First time point:", signal[0])
print("Minimum voltage:", np.min(signal))
print("Maximum voltage:", np.max(signal))
print("Missing values:", np.isnan(signal).sum())