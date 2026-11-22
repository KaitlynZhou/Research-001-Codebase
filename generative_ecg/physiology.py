"""Fixed beat regions and linear limb-lead relationships."""

import numpy as np

from .beats import FS, PRE, SHAPE

REGIONS_MS = {"P": (-260, -90), "QRS": (-90, 110), "T": (110, 440)}
LIMB_CONSTRAINTS = np.array([
    [1, -1, 1, 0, 0, 0],
    [0.5, 0.5, 0, 1, 0, 0],
    [-1, 0.5, 0, 0, 1, 0],
    [0.5, -1, 0, 0, 0, 1],
], dtype=np.float32)


def region_slice(bounds_ms):
    start, end = bounds_ms
    return slice(PRE + round(start * FS / 1000), PRE + round(end * FS / 1000))


def reconstruction_weights(p_weight, t_weight):
    weights = np.ones(SHAPE[-1], dtype=np.float32)
    weights[region_slice(REGIONS_MS["P"])] = p_weight
    weights[region_slice(REGIONS_MS["T"])] = t_weight
    return weights / weights.mean()


def project_limb_leads(wave):
    """Return the closest ECG satisfying all four linear limb identities."""
    constraints = LIMB_CONSTRAINTS.astype(float)
    projection = np.eye(6) - np.linalg.pinv(constraints) @ constraints
    result = wave.copy()
    result[..., :6, :] = projection @ wave[..., :6, :]
    return result
