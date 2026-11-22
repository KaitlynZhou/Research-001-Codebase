"""Train and benchmark a lead-II interval detector using annotated beat segments."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier

from .beats import FS, PRE, SHAPE
from .validate_intervals import benchmark, windows
from .delineation import lead_aware_intervals
from .evaluate import morphology


def features(beats):
    lead = beats[:, 1].astype(np.float32)
    magnitude = np.sqrt(np.mean(beats**2, axis=1))
    time = np.broadcast_to((np.arange(SHAPE[-1]) - PRE) / FS, lead.shape)
    columns = [time]
    for wave in (lead, magnitude):
        for width in (2, 6, 15):
            smooth = gaussian_filter1d(wave, width, axis=-1)
            columns.extend((smooth, gaussian_filter1d(wave, width, order=1, axis=-1) * width,
                            gaussian_filter1d(wave, width, order=2, axis=-1) * width**2))
        columns.append(wave - gaussian_filter1d(wave, 25, axis=-1))
    return np.stack(columns, axis=-1)


def best_segment(probability, minimum, start, stop):
    section = probability[start:stop]
    edges = np.flatnonzero(np.diff(np.r_[False, section >= 0.55, False]))
    candidates = [(lo, hi) for lo, hi in edges.reshape(-1, 2)
                  if hi - lo >= minimum and lo > 0 and hi < len(section)]
    if not candidates:
        return None
    lo, hi = max(candidates, key=lambda pair: section[slice(*pair)].sum())
    return start + lo, start + hi - 1


def segment_amplitude(signal, segment):
    """Peak deflection from the line joining the segment endpoints, in mV."""
    lo, hi = segment
    wave = signal[lo:hi + 1]
    return float(np.max(np.abs(wave - np.linspace(wave[0], wave[-1], len(wave)))))


class AnnotatedDetector:
    def __init__(self, root):
        training = [(x, labels) for _, x, _, labels in windows(
            root, with_segments=True, records=set(range(1, 81, 2)))]
        if not training:
            raise ValueError("Missing odd-numbered LUDB training records")
        beats, labels = map(np.stack, zip(*training))
        x = features(beats)
        smoothed = gaussian_filter1d(beats[:, 1], 3, axis=-1)
        self.floor = {}
        for wave_class in (1, 3):
            amplitudes = []
            for wave, mask in zip(smoothed, labels):
                indices = np.flatnonzero(mask == wave_class)
                if len(indices) >= 15:
                    amplitudes.append(segment_amplitude(wave, (indices[0], indices[-1])))
            self.floor[wave_class] = float(np.quantile(amplitudes, 0.10))
        self.classifier = HistGradientBoostingClassifier(
            max_iter=90, max_leaf_nodes=23, min_samples_leaf=60, l2_regularization=3,
            early_stopping=False, random_state=42).fit(x.reshape(-1, x.shape[-1]), labels.ravel())

    def measure(self, beats, *, evidence=True):
        x = features(beats)
        probability = self.classifier.predict_proba(x.reshape(-1, x.shape[-1])).reshape(
            len(beats), SHAPE[-1], 4)
        probability = gaussian_filter1d(probability, 2, axis=1)
        smoothed = gaussian_filter1d(beats[:, 1], 3, axis=-1)
        estimates = np.full((len(beats), 3), np.nan)
        for idx, (p_prob, qrs_prob, t_prob) in enumerate(probability[:, :, 1:].transpose(0, 2, 1)):
            qrs = best_segment(qrs_prob, 18, PRE - 75, PRE + 75)
            if qrs is None:
                continue
            onset, offset = qrs
            estimates[idx, 1] = offset - onset
            p = best_segment(p_prob, 15, 3, onset)
            t = best_segment(t_prob, 22, offset + 1, SHAPE[-1] - 3)
            if p is not None and (not evidence or segment_amplitude(smoothed[idx], p) >= self.floor[1]):
                estimates[idx, 0] = onset - p[0]
            if t is not None and (not evidence or segment_amplitude(smoothed[idx], t) >= self.floor[3]):
                estimates[idx, 2] = t[1] - onset
        return estimates * (1000 / FS)

    def __call__(self, beat):
        return self.measure(beat[None])[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ludb_data", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--confirmation", type=Path)
    args = parser.parse_args()
    detector = AnnotatedDetector(args.ludb_data)
    report = benchmark(args.ludb_data, {"original": morphology, "lead_aware": lead_aware_intervals,
                                      "annotated": detector,
                                      "annotated_ungated": lambda beat: detector.measure(
                                          beat[None], evidence=False)[0]},
                       confirmation=args.confirmation)
    report["evidence_floor_mv"] = {"P": detector.floor[1], "T": detector.floor[3]}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
