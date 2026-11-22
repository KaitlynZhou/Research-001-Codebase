"""Lead-aware interval estimates for aligned median-beat ECGs."""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

from .beats import FS, PRE


def wave_boundary(leads, start, stop, *, end):
    section = leads[:, start:stop]
    if section.shape[-1] < 30:
        return np.nan
    estimates = []
    for wave in section:
        baseline = np.linspace(np.median(wave[:8]), np.median(wave[-8:]), len(wave))
        magnitude = np.abs(wave - baseline)
        peaks, properties = find_peaks(magnitude, prominence=0.02 if not end else 0.03,
                                      width=(8, None), distance=12)
        eligible = (peaks >= 10) & (peaks < len(wave) - 10)
        if not eligible.any():
            continue
        peak = peaks[np.argmax(properties["prominences"] * eligible)]
        _, _, left, right = peak_widths(magnitude, [peak], rel_height=0.9)
        if left[0] < 3 or right[0] > len(wave) - 4:
            continue
        estimates.append(start + (right[0] if end else left[0]))
    if not estimates:
        return np.nan
    return float(np.median(estimates))


def lead_aware_intervals(beat):
    signals = gaussian_filter1d(beat, sigma=2.5, axis=-1)
    qrs_activity = np.sqrt(np.mean(signals**2, axis=0))
    search = qrs_activity[PRE - 30:PRE + 31]
    peak = PRE - 30 + np.argmax(search)
    if qrs_activity[peak] < 0.05:
        return np.full(3, np.nan)
    level = max(0.025, 0.08 * qrs_activity[peak])
    onset = offset = peak
    while onset > PRE - 65 and qrs_activity[onset] > level:
        onset -= 1
    while offset < PRE + 65 and qrs_activity[offset] > level:
        offset += 1
    if onset == PRE - 65 or offset == PRE + 65 or offset - onset < 17:
        return np.full(3, np.nan)
    p = wave_boundary(signals[[1, 0, 10]], PRE - 132, onset - 4, end=False)
    t = wave_boundary(signals[[1, 7, 10]], offset + 12, len(qrs_activity) - 6, end=True)
    return np.array([onset - p, offset - onset, t - onset]) * (1000 / FS)
