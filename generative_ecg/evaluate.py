"""Compare held-out median beats, reconstructions and upstream latent samples."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance


from .beats import FS, PRE, plot_grid


def morphology(beat):
    """Measure threshold-based timing proxies on a multilead activity envelope."""
    activity = np.sqrt(np.mean(beat**2, axis=0))
    activity = np.convolve(activity, np.ones(9) / 9, mode="same")
    qrs = activity[PRE - 35:PRE + 36]
    peak = PRE - 35 + np.argmax(qrs)
    level = 0.15 * activity[peak]
    onset, offset = peak, peak
    while onset > PRE - 60 and activity[onset] > level:
        onset -= 1
    while offset < PRE + 60 and activity[offset] > level:
        offset += 1

    def wave_edges(start, end):
        section = activity[start:end]
        if len(section) < 10:
            return np.nan, np.nan
        best = np.argmax(section)
        baseline = np.percentile(section, 10)
        prominence = section[best] - baseline
        if prominence < 0.015 or best < 3 or best >= len(section) - 3:
            return np.nan, np.nan
        left, right = best, best
        level = baseline + 0.15 * prominence
        while left > 0 and section[left] > level:
            left -= 1
        while right < len(section) - 1 and section[right] > level:
            right += 1
        if left == 0 or right == len(section) - 1:
            return np.nan, np.nan
        return start + left, start + right

    if onset == PRE - 60 or offset == PRE + 60 or activity[peak] < 0.05:
        return [np.nan] * 3
    p_start, _ = wave_edges(PRE - 130, onset - 6)
    _, t_end = wave_edges(offset + 15, len(activity) - 5)
    return np.array([onset - p_start, offset - onset, t_end - onset]) * 1000 / FS


def distribution(x):
    values = np.asarray(x)
    values = values[np.isfinite(values)]
    return dict(n=int(len(values)), minimum=float(np.min(values)) if len(values) else None,
                median=float(np.median(values)) if len(values) else None,
                p05=float(np.percentile(values, 5)) if len(values) else None,
                p95=float(np.percentile(values, 95)) if len(values) else None)


def limb_errors(x):
    i, ii, iii, avr, avl, avf = np.moveaxis(x[:, :6], 1, 0)
    residual = np.stack((iii - (ii - i), avr + (i + ii) / 2,
                         avl - (i - ii / 2), avf - (ii - i / 2)), axis=1)
    return np.sqrt(np.mean(residual**2, axis=(1, 2)))


def nearest(query, train):
    norms = np.einsum("ij,ij->i", train, train)
    distances = []
    for start in range(0, len(query), 32):
        batch = query[start:start + 32]
        batch_norms = np.einsum("ij,ij->i", batch, batch)
        squared = batch_norms[:, None] + norms - 2 * (batch @ train.T)
        distances.extend(np.sqrt(np.maximum(0, squared.min(axis=1)) / train.shape[1]))
    return np.asarray(distances)


def evaluate(data, generated, out):
    real = np.load(data / "test.npy")
    train = np.load(data / "train.npy", mmap_mode="r")
    recon = np.load(generated / "reconstructed.npy")
    synthetic = np.load(generated / "synthetic.npy")
    if recon.shape != real.shape or not all(np.isfinite(x).all() for x in (real, recon, synthetic)):
        raise ValueError("Invalid evaluation arrays")
    out.mkdir(parents=True, exist_ok=True)
    groups = dict(real=real, reconstructed=recon, synthetic=synthetic)
    comparison = {}
    timings = {name: np.asarray([morphology(beat) for beat in x]) for name, x in groups.items()}
    for name, x in groups.items():
        comparison[name] = {
            "count": len(x),
            "limb_lead_identity_rmse_mv": distribution(limb_errors(x)),
            "peak_to_peak_mv": [distribution(a) for a in np.ptp(x, axis=-1).T],
            "timing_proxies_ms": {label: distribution(values) for label, values in
                                  zip(("PR", "QRS", "QT"), timings[name].T)},
        }
    rng = np.random.default_rng(42)
    train_flat = train.reshape(len(train), -1)
    diversity = {}
    for name in ("real", "synthetic"):
        x = groups[name]
        chosen = x[rng.choice(len(x), min(500, len(x)), replace=False)].reshape(-1, train_flat.shape[1])
        pairwise = pdist(chosen) / np.sqrt(chosen.shape[1])
        diversity[name] = {
            "subset_size": len(chosen),
            "pairwise_rmse_mv": distribution(pairwise),
            "pair_fraction_below_0_01_mv": float(np.mean(pairwise < 0.01)) if len(pairwise) else None,
            "nearest_training_rmse_mv": distribution(nearest(chosen, train_flat)),
            "mean_pointwise_sd_mv": float(np.std(x, axis=0).mean()),
            "exact_duplicates_full_set": int(len(x) - len({beat.tobytes() for beat in x})),
        }
    interval_comparison = {}
    for index, label in enumerate(("PR", "QRS", "QT")):
        measured = {name: x[:, index] for name, x in timings.items()}
        observed = {name: x[np.isfinite(x)] for name, x in measured.items()}
        interval_comparison[label] = dict(
            paired_reconstruction_absolute_error_ms=distribution(
                abs(measured["real"] - measured["reconstructed"])),
            real_synthetic_wasserstein_ms=float(wasserstein_distance(
                observed["real"], observed["synthetic"]))
            if len(observed["real"]) and len(observed["synthetic"]) else None,
        )
    result = dict(
        reconstruction_rmse_mv=float(np.sqrt(np.mean((recon - real)**2))),
        reconstruction_mae_mv=float(np.mean(np.abs(recon - real))),
        reconstruction_per_lead_rmse_mv=np.sqrt(np.mean((recon - real)**2, axis=(0, 2))).tolist(),
        comparison=comparison, diversity=diversity, interval_comparison=interval_comparison,
    )
    (out / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    for idx in range(min(3, len(real), len(synthetic))):
        plot_grid(out / f"comparison_{idx + 1}.png",
                  {"Real": real[idx], "Reconstruction": recon[idx], "Synthetic": synthetic[idx]})
    plot_grid(out / "mean_waveforms.png", {name: x.mean(axis=0) for name, x in groups.items()})
    plot_grid(out / "waveform_variation.png", {name: x.std(axis=0) for name, x in groups.items()},
              ylabel="Pointwise standard deviation (mV)")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for index, (axis, label) in enumerate(zip(axes, ("PR", "QRS", "QT"))):
        for name, values in timings.items():
            present = values[:, index][np.isfinite(values[:, index])]
            if len(present):
                axis.hist(present, bins=np.arange(0, 701, 10), density=True,
                          histtype="step", label=name)
        axis.set_xlabel(f"{label} proxy (ms)")
    axes[0].set_ylabel("Density")
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(out / "intervals.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "generated", "out"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    evaluate(args.data, args.generated, args.out)
