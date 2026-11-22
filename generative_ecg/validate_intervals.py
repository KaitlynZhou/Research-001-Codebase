"""Benchmark interval estimates on native 500 Hz annotated LUDB beat windows."""

import argparse
import json
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import sosfiltfilt

from .beats import BASELINE, FS, LEADS, POST, PRE
from .evaluate import morphology
from .delineation import lead_aware_intervals

INTERVALS = ("PR", "QRS", "QT")


def annotated_waves(annotation):
    points = zip(annotation.sample, annotation.symbol)
    start, peak = None, None
    for sample, symbol in points:
        if symbol == "(":
            start, peak = sample, None
        elif symbol in ("p", "N", "t"):
            peak = (sample, symbol)
        elif symbol == ")":
            if start is not None and peak is not None and start < peak[0] < sample:
                yield (start, peak[0], sample, peak[1])
            start, peak = None, None


def windows(root, *, with_segments=False, records=None):
    for header in sorted(root.glob("*.hea"), key=lambda p: int(p.stem)):
        record_id = int(header.stem)
        if records is not None and record_id not in records:
            continue
        record = wfdb.rdrecord(str(header.with_suffix("")))
        if record.fs != FS or record.p_signal.shape != (5000, 12):
            raise ValueError(f"Unexpected LUDB record format in {header}")
        names = [name.upper() for name in record.sig_name]
        wave = record.p_signal[:, [names.index(lead) for lead in LEADS]].T
        wave = sosfiltfilt(BASELINE, wave, axis=-1)
        annotations = list(annotated_waves(wfdb.rdann(str(header.with_suffix("")), "ii")))
        for i, (onset, anchor, offset, kind) in enumerate(annotations):
            if kind != "N" or anchor < PRE or anchor + POST > wave.shape[-1]:
                continue
            prev = annotations[i - 1] if i else None
            following = annotations[i + 1] if i + 1 < len(annotations) else None
            pr = onset - prev[0] if prev and prev[3] == "p" and prev[0] >= anchor - PRE else np.nan
            qt = following[2] - onset if following and following[3] == "t" and following[2] < anchor + POST else np.nan
            cropped = wave[:, anchor - PRE:anchor + POST]
            intervals = np.array([pr, offset - onset, qt]) * (1000 / FS)
            if not with_segments:
                yield record_id, cropped, intervals
                continue
            labels = np.zeros(PRE + POST, dtype=np.int8)
            start = anchor - PRE
            labels[max(onset - start, 0):min(offset - start + 1, len(labels))] = 2
            if np.isfinite(pr):
                labels[prev[0] - start:prev[2] - start + 1] = 1
            if np.isfinite(qt):
                labels[following[0] - start:following[2] - start + 1] = 3
            yield record_id, cropped, intervals, labels


def benchmark(root, detectors, confirmation=None):
    references = list(windows(root))
    if confirmation is not None:
        references.extend(windows(confirmation))
    if confirmation is not None and not any(record > 80 for record, _, _ in references):
        raise ValueError("No independent confirmation records found")
    if not references:
        raise ValueError("No LUDB WFDB headers found")
    report = {}
    assignments = {"odd": set(range(1, 81, 2)), "even": set(range(2, 81, 2))}
    external = {record for record, _, _ in references if record > 80}
    if external:
        assignments["independent"] = external
    for split, records in assignments.items():
        rows = [(wave, truth) for record, wave, truth in references if record in records]
        if not rows:
            continue
        truth = np.stack([item[1] for item in rows])
        report[split] = {"beats": len(rows)}
        for name, detector in detectors.items():
            estimate = np.asarray([detector(item[0]) for item in rows])
            metrics = {}
            for col, label in enumerate(INTERVALS):
                valid = np.isfinite(truth[:, col])
                measured = np.isfinite(estimate[:, col])
                match = valid & measured
                error = estimate[match, col] - truth[match, col]
                metrics[label] = dict(reference=int(valid.sum()), matched=int(match.sum()),
                                      detection_without_in_window_reference=int((measured & ~valid).sum()),
                                      mae_ms=float(np.abs(error).mean()) if len(error) else None,
                                      bias_ms=float(error.mean()) if len(error) else None,
                                      within_30ms=int((np.abs(error) <= 30).sum()))
            report[split][name] = metrics
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ludb_data", type=Path)
    parser.add_argument("--original-only", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    detectors = {"original": morphology}
    if not args.original_only:
        detectors["lead_aware"] = lead_aware_intervals
    result = json.dumps(benchmark(args.ludb_data, detectors), indent=2)
    if args.out:
        args.out.write_text(result + "\n")
    print(result)
