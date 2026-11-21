"""Extract aligned, 12-lead PTB-XL median beats at native resolution."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import wfdb
from scipy import signal

FS = 500
LEADS = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
PRE, POST = 150, 250
SHAPE = (12, PRE + POST)
BANDPASS = signal.butter(3, (8, 30), fs=FS, btype="bandpass", output="sos")
BASELINE = signal.butter(2, 0.5, fs=FS, btype="highpass", output="sos")


def read_record(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.parts[:1] != ("records500",):
        raise ValueError("Expected a relative records500 path")
    wave, fields = wfdb.rdsamp(str(root / path))
    names = [name.upper() for name in fields["sig_name"]]
    if (fields["fs"] != FS or wave.shape != (5000, 12) or set(names) != set(LEADS)
            or any(unit.lower() != "mv" for unit in fields["units"])):
        raise ValueError(f"Invalid waveform contract at {path}")
    return wave[:, [names.index(name) for name in LEADS]].T


def extract(wave):
    """Return the median, refined QRS anchors, and accepted aligned beats."""
    if wave.shape != (12, 5000) or not np.isfinite(wave).all():
        raise ValueError("invalid_waveform")
    qrs = signal.sosfiltfilt(BANDPASS, wave, axis=-1)
    power = np.sqrt(np.mean(qrs**2, axis=0))
    envelope = np.convolve(power, np.ones(35) / 35, mode="same")
    reference = np.percentile(envelope, 97)
    if reference < 0.015:
        raise ValueError("weak_qrs")
    peaks, _ = signal.find_peaks(envelope, distance=125, height=0.22 * reference,
                                 prominence=0.16 * reference)
    candidates = peaks[(peaks >= PRE + 20) & (peaks < wave.shape[-1] - POST - 20)]
    refined = np.array([p - 20 + np.argmax(power[p - 20:p + 21])
                        for p in candidates], dtype=int)
    cleaned = signal.sosfiltfilt(BASELINE, wave, axis=-1)
    if len(refined) < 3:
        raise ValueError("too_few_complete_beats")
    windows = np.stack([cleaned[:, p - PRE:p + POST] for p in refined])
    valid = np.max(np.ptp(windows, axis=-1), axis=1) < 12
    windows, refined = windows[valid], refined[valid]
    if len(windows) < 3:
        raise ValueError("too_few_valid_beats")
    core = windows[:, :, PRE - 55:PRE + 55]
    template = np.median(core, axis=0)
    residual = np.sqrt(np.mean((core - template)**2, axis=(1, 2)))
    limit = max(0.08, np.median(residual) + 3 * np.median(abs(residual - np.median(residual))))
    keep = residual <= limit
    if keep.sum() < 3:
        raise ValueError("too_few_consistent_beats")
    return np.median(windows[keep], axis=0).astype(np.float32), refined[keep], windows[keep]


def plot_grid(path, traces, ylabel="Amplitude (mV)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 3, figsize=(12, 8), sharex=True, sharey=True)
    time = (np.arange(SHAPE[1]) - PRE) * 1000 / FS
    for i, ax in enumerate(axes.flat):
        for label, wave in traces.items():
            ax.plot(time, wave[i], label=label, lw=0.9, alpha=0.65)
        ax.set_title(LEADS[i])
        ax.axvline(0, color="0.75", lw=0.6)
    axes.flat[0].legend(fontsize=8)
    fig.supxlabel("Time from QRS anchor (ms)")
    fig.supylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def prepare(root, out):
    with (root / "ptbxl_database.csv").open() as file:
        rows = list(csv.DictReader(file))
    patients = {}
    cohorts = {name: [] for name in ("train", "val", "test")}
    if len({row["ecg_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate ECG identifiers")
    for row in rows:
        fold = int(row["strat_fold"])
        if fold not in range(1, 11) or not row["patient_id"]:
            raise ValueError("Missing patient or invalid fold")
        if patients.setdefault(row["patient_id"], fold) != fold:
            raise ValueError("Patient crosses official folds")
        cohorts["train" if fold < 9 else "val" if fold == 9 else "test"].append(row)
    out.mkdir(parents=True)
    summary = {}
    for split, cohort in cohorts.items():
        beats, metadata = [], []
        for row in cohort:
            wave = read_record(root, row["filename_hr"])
            record = {key: row[key] for key in ("ecg_id", "patient_id", "strat_fold", "scp_codes")}
            try:
                median, _, aligned = extract(wave)
            except ValueError as exc:
                metadata.append(dict(record, status=str(exc), array_index="", accepted_beats=0))
                continue
            if not beats and split == "train":
                plot_grid(out / "extraction_example.png",
                          dict(Beat=aligned[0], Median=median))
            metadata.append(dict(record, status="accepted", array_index=len(beats),
                                 accepted_beats=len(aligned)))
            beats.append(median)
        np.save(out / f"{split}.npy", np.stack(beats))
        with (out / f"{split}.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(metadata[0]))
            writer.writeheader()
            writer.writerows(metadata)
        summary[split] = dict(Counter(row["status"] for row in metadata))
        print(split, summary[split], flush=True)
    (out / "contract.json").write_text(json.dumps(
        dict(fs=FS, leads=LEADS, units="mV", shape=SHAPE, anchor=PRE, counts=summary), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    prepare(args.root, args.out)
