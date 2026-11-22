from types import SimpleNamespace

import numpy as np

from generative_ecg.beats import SHAPE
from generative_ecg.delineation import lead_aware_intervals
from generative_ecg.evaluate import limb_errors
from generative_ecg.physiology import REGIONS_MS, project_limb_leads, reconstruction_weights, region_slice
from generative_ecg.validate_intervals import annotated_waves


def test_reconstruction_weights_keep_unit_mean_and_target_regions():
    weights = reconstruction_weights(3, 2)
    p, qrs, t = (weights[region_slice(bounds)] for bounds in REGIONS_MS.values())
    np.testing.assert_allclose(weights.mean(), 1)
    np.testing.assert_allclose(p.mean() / qrs.mean(), 3, rtol=1e-6)
    np.testing.assert_allclose(t.mean() / qrs.mean(), 2, rtol=1e-6)


def test_projection_removes_limb_residual_and_preserves_chest_leads():
    beats = np.random.default_rng(42).normal(size=(4, *SHAPE)).astype(np.float32)
    corrected = project_limb_leads(beats)
    np.testing.assert_array_equal(corrected[:, 6:], beats[:, 6:])
    np.testing.assert_allclose(limb_errors(corrected), 0, atol=1e-7)
    np.testing.assert_allclose(project_limb_leads(corrected), corrected, atol=3e-7)
    assert np.all(limb_errors(beats) > 1)


def test_annotation_parser_keeps_only_complete_waves():
    annotation = SimpleNamespace(sample=[0, 1, 2, 3, 4, 5, 6, 7],
                                 symbol=["p", "(", "p", ")", "(", "N", "(", ")"])
    assert list(annotated_waves(annotation)) == [(1, 2, 3, "p")]
    assert np.isnan(lead_aware_intervals(np.zeros(SHAPE))).all()
