"""Generate 12-lead median beats with a joint mixture in waveform PCA space."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from .beats import SHAPE
from .physiology import LIMB_CONSTRAINTS, reconstruction_weights

LIMB_BASIS = np.linalg.svd(LIMB_CONSTRAINTS.astype(np.float64), full_matrices=True)[2][-2:].T


def independent_leads(beats):
    return np.concatenate((LIMB_BASIS.T @ beats[:, :6], beats[:, 6:]), axis=1)


def restore_leads(beats):
    return np.concatenate((LIMB_BASIS @ beats[:, :2], beats[:, 2:]), axis=1)


def fit_waveforms(beats, *, rank=64, components=12, p_weight=6, t_weight=3, seed=42):
    if beats.ndim != 3 or beats.shape[1:] != SHAPE or not np.isfinite(beats).all():
        raise ValueError("Expected finite 12-lead training beats")
    if (not np.isfinite([p_weight, t_weight]).all() or min(rank, components, p_weight, t_weight) <= 0
            or rank > min(len(beats), 8 * SHAPE[-1]) or components > len(beats)):
        raise ValueError("Invalid waveform model configuration")
    weights = np.sqrt(reconstruction_weights(p_weight, t_weight))
    # Float32 is sufficient for PCA. Estimate mixture covariance in float64.
    independent = independent_leads(beats).astype(np.float32)
    matrix = (independent * weights).reshape(len(beats), -1)
    pca = PCA(n_components=rank, whiten=True, svd_solver="randomized", random_state=seed)
    scores = pca.fit_transform(matrix).astype(np.float64)
    if np.any(pca.explained_variance_ <= 0):
        raise ValueError("PCA rank exceeds the number of varying directions")
    mixture = GaussianMixture(n_components=components, covariance_type="full", reg_covar=0.025,
                              max_iter=250, n_init=2, random_state=seed).fit(scores)
    if not mixture.converged_:
        raise RuntimeError("Waveform mixture did not converge")
    return weights, pca, mixture


def decode(scores, weights, mean, scale, components):
    independent = ((scores * scale) @ components + mean).reshape(-1, 8, SHAPE[-1])
    beats = restore_leads(independent / weights).astype(np.float32)
    if not np.isfinite(beats).all():
        raise ValueError("Non-finite decoded beats")
    return beats


def draw(model, count, seed):
    """Sample from saved mixture parameters without loading serialized Python objects."""
    if count <= 0:
        raise ValueError("Sample count must be positive")
    rng = np.random.default_rng(seed)
    labels = rng.choice(len(model["mixture_weights"]), count, p=model["mixture_weights"])
    means = model["mixture_means"]
    scores = np.empty((count, means.shape[1]))
    for component in np.unique(labels):
        positions = labels == component
        factor = np.linalg.cholesky(model["mixture_covariances"][component])
        scores[positions] = means[component] + rng.standard_normal(
            (int(positions.sum()), means.shape[1])) @ factor.T
    return decode(scores, *(model[key] for key in ("weights", "mean", "scale", "components")))


def reconstruct(model, beats):
    weights = model["weights"]
    matrix = (independent_leads(beats).astype(np.float32) * weights).reshape(len(beats), -1)
    scores = (matrix - model["mean"]) @ model["components"].T / model["scale"]
    return decode(scores, *(model[key] for key in ("weights", "mean", "scale", "components")))


def run(data, out, args):
    train = np.load(data / "train.npy", mmap_mode="r")
    fitting = train
    target_split = args.reconstruct
    if args.development:
        with (data / "train.csv").open() as handle:
            records = [r for r in csv.DictReader(handle) if r["status"] == "accepted"]
        fit = [int(r["array_index"]) for r in records if int(r["strat_fold"]) < 8]
        dev = [int(r["array_index"]) for r in records if int(r["strat_fold"]) == 8]
        fitting, target = train[fit], train[dev]
        target_split = "train_fold_8"
    elif target_split:
        target = np.load(data / f"{target_split}.npy", mmap_mode="r")
    weights, pca, model = fit_waveforms(fitting, rank=args.rank, components=args.components,
                                        p_weight=args.p_weight, t_weight=args.t_weight, seed=args.seed)
    parameters = dict(weights=weights, mean=pca.mean_, scale=np.sqrt(pca.explained_variance_),
                      components=pca.components_, mixture_weights=model.weights_,
                      mixture_means=model.means_, mixture_covariances=model.covariances_)
    out.mkdir(parents=True)
    np.savez(out / "model.npz", **parameters)
    np.save(out / "synthetic.npy", draw(parameters, args.count, args.seed))
    if target_split:
        np.save(out / "reconstructed.npy", reconstruct(parameters, target))
        if args.development:
            np.save(out / "reference.npy", target)
    config = dict(fitting_folds=list(range(1, 8 if args.development else 9)),
                  training_beats=len(fitting), reconstruction_split=target_split,
                  rank=args.rank, components=args.components, p_weight=args.p_weight,
                  t_weight=args.t_weight, seed=args.seed, count=args.count,
                  variance_explained=float(pca.explained_variance_ratio_.sum()))
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--components", type=int, default=12)
    parser.add_argument("--p-weight", type=float, default=6)
    parser.add_argument("--t-weight", type=float, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--reconstruct", choices=("val", "test"))
    args = parser.parse_args()
    if args.development and args.reconstruct:
        parser.error("Development reconstructs only the reserved training fold")
    run(args.data, args.out, args)
