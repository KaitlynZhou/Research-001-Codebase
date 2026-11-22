"""Compare frozen generators with a shared annotated-detector and waveform audit."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance

from .annotated_detector import AnnotatedDetector
from .beats import FS, LEADS, PRE, plot_grid
from .evaluate import distribution, limb_errors, morphology, nearest
from .physiology import REGIONS_MS, region_slice
from .validate_intervals import INTERVALS
from .waveform_model import reconstruct


def waveform_features(beats):
    smoothed = gaussian_filter1d(beats, 3, axis=-1)
    regions = {}
    for name, bounds in REGIONS_MS.items():
        region = smoothed[..., region_slice(bounds)]
        chord = region[..., :1] + (region[..., -1:] - region[..., :1]) * np.linspace(
            0, 1, region.shape[-1])
        regions[name] = np.max(np.abs(region - chord), axis=-1)
    return regions, np.sqrt(np.mean((beats - smoothed)**2, axis=(1, 2)))


def interval_summary(values, reference):
    result = {}
    for name, estimate, real in zip(INTERVALS, values.T, reference.T):
        present = estimate[np.isfinite(estimate)]
        actual = real[np.isfinite(real)]
        result[name] = {**distribution(present), "coverage": float(len(present) / len(estimate)),
                        "distance_to_real_ms": float(wasserstein_distance(present, actual))
                        if len(present) and len(actual) else None}
    return result


def audit(real, train, groups, reconstructions, detector):
    for name, beats in {"real": real, **groups}.items():
        if beats.ndim != 3 or beats.shape[1:] != real.shape[1:] or not len(beats):
            raise ValueError(f"Invalid waveform array for {name}")
        if not np.isfinite(beats).all():
            raise ValueError(f"Non-finite waveform array for {name}")
    for name, beats in reconstructions.items():
        if beats.shape != real.shape or not np.isfinite(beats).all():
            raise ValueError(f"Invalid paired reconstruction array for {name}")
    reference = detector.measure(real)
    original_reference = np.array([morphology(beat) for beat in real])
    real_regions, _ = waveform_features(real)
    train_flat = train.reshape(len(train), -1)
    result = {"groups": {}, "reconstructions": {}}
    for label, beats in {"real": real, **groups}.items():
        rng = np.random.default_rng(42)
        if label != "real":
            rng.choice(len(real), min(500, len(real)), replace=False)
        sample = beats[rng.choice(len(beats), min(500, len(beats)), replace=False)]
        sample = sample.reshape(len(sample), -1)
        pair = pdist(sample) / np.sqrt(sample.shape[1])
        region, residual = waveform_features(beats)
        result["groups"][label] = dict(
            count=len(beats), annotated_intervals=interval_summary(detector.measure(beats), reference),
            original_intervals=interval_summary(np.array([morphology(beat) for beat in beats]),
                                                original_reference),
            pairwise_rmse_mv=distribution(pair),
            near_pair_fraction=float(np.mean(pair < 0.01)) if len(pair) else None,
            nearest_training_rmse_mv=distribution(nearest(sample, train_flat)),
            exact_duplicates=len(beats) - len({beat.tobytes() for beat in beats}),
            limb_error_mv=distribution(limb_errors(beats)),
            fast_residual_rmse_mv=distribution(residual),
            regional_deflection_mv={
                name: {lead: distribution(r[:, i]) for i, lead in enumerate(LEADS)}
                for name, r in region.items()},
            mean_regional_deflection_distance_mv={
                name: float(np.mean([wasserstein_distance(a, b)
                                     for a, b in zip(r.T, real_regions[name].T)]))
                for name, r in region.items()})
    for label, beats in reconstructions.items():
        difference = beats - real
        timing = detector.measure(beats)
        result["reconstructions"][label] = dict(
            rmse_mv=float(np.sqrt(np.mean(difference**2))),
            region_rmse_mv={name: float(np.sqrt(np.mean(difference[..., region_slice(bounds)]**2)))
                            for name, bounds in REGIONS_MS.items()},
            annotated_intervals=interval_summary(timing, reference),
            paired_interval_absolute_error_ms={name: distribution(abs(a - b))
                                               for name, a, b in zip(INTERVALS, timing.T, reference.T)})
    return result


def main(args):
    root = args.artifacts
    data = root / "beats"
    path = args.waveform
    train = np.load(data / "train.npy", mmap_mode="r")
    real = np.load(data / f"{args.split}.npy")
    locations = {"original": root / "generated", "frozen_mixture": root / "mixture_generated",
                 "weighted_vae": root / "physiology_p_test", "waveform_mixture": path}
    groups = {name: np.load(directory / "synthetic.npy")
              for name, directory in locations.items() if (directory / "synthetic.npy").is_file()}
    if "waveform_mixture" not in groups:
        raise ValueError("Missing waveform mixture samples")
    with np.load(path / "model.npz", allow_pickle=False) as model:
        recon = {"waveform_mixture": reconstruct(model, real)}
    if args.split == "test":
        for name in ("original", "weighted_vae"):
            file = locations[name] / "reconstructed.npy"
            if file.is_file():
                recon[name] = np.load(file)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    report = audit(real, train, groups, recon, AnnotatedDetector(args.ludb))
    report["split"] = args.split
    (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    names = {"original": "Original", "frozen_mixture": "Frozen mixture",
             "weighted_vae": "Weighted VAE", "waveform_mixture": "Waveform mixture"}
    display = {"Real": real, **{names[name]: wave for name, wave in groups.items()}}
    plot_grid(out / "waveform_spread.png", {name: wave.std(axis=0) for name, wave in display.items()},
              ylabel="Pointwise standard deviation (mV)")
    for index in range(min(3, *(len(wave) for wave in display.values()))):
        plot_grid(out / f"example_{index + 1}.png", {name: wave[index] for name, wave in display.items()})
    plot_grid(out / "reconstruction.png", {"Real": real[0],
              **{names[name]: wave[0] for name, wave in recon.items()}})
    plot_p_region(out / "p_wave_examples.png", display)


def plot_p_region(path, groups):
    import matplotlib.pyplot as plt

    region = region_slice(REGIONS_MS["P"])
    time = (np.arange(region.start, region.stop) - PRE) * 1000 / FS
    fig, axes = plt.subplots(len(groups), 1, figsize=(8, 2 * len(groups)),
                            sharex=True, sharey=True, squeeze=False)
    for axis, (label, beats) in zip(axes[:, 0], groups.items()):
        for wave in beats[:6, LEADS.index("II"), region]:
            axis.plot(time, wave, alpha=0.7, linewidth=1)
        axis.set_title(label, loc="left")
        axis.axhline(0, color="0.75", linewidth=0.6)
    fig.supxlabel("Time from QRS anchor (ms)")
    fig.supylabel("Lead II amplitude (mV)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifacts", "waveform", "ludb", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    main(parser.parse_args())
