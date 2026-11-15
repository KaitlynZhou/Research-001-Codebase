
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from pathlib import Path


df = pd.read_csv("data/processed/dataset1/metadata.csv")


record = df[df["CLBBB"] == 1].iloc[0]


record_path = Path("data/raw/ptb-xl") / record["filename_lr"]
signal, fields = wfdb.rdsamp(str(record_path))


time = np.arange(signal.shape[0]) / fields["fs"]


fig, axes = plt.subplots(12, 1, figsize=(12, 16), sharex=True)

for i, lead in enumerate(fields["sig_name"]):
    axes[i].plot(time, signal[:, i], linewidth=0.8)
    axes[i].set_ylabel(lead)
    axes[i].grid(alpha=0.3)

axes[-1].set_xlabel("Time (seconds)")
fig.suptitle(f"12-Lead ECG — CLBBB (ECG ID: {record['ecg_id']})")
plt.tight_layout()
plt.show()