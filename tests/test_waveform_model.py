import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from generative_ecg.annotated_detector import best_segment, segment_amplitude
from generative_ecg.evaluate import limb_errors
from generative_ecg.waveform_model import draw, fit_waveforms, reconstruct, restore_leads


def test_waveform_model_roundtrip_and_sampling():
    rng = np.random.default_rng(4)
    x = np.arange(400)
    shape = np.exp(-((x - 150) / 20)**2)
    p_wave = np.exp(-((x - 75) / 12)**2)
    independent = (rng.normal(size=(80, 8, 1)) * shape
                   + rng.normal(size=(80, 8, 1)) * p_wave / 4)
    beats = restore_leads(independent).astype(np.float32)
    weights, pca, mixture = fit_waveforms(beats, rank=16, components=2)
    parameters = dict(weights=weights, mean=pca.mean_, scale=pca.explained_variance_**0.5,
                      components=pca.components_, mixture_weights=mixture.weights_,
                      mixture_means=mixture.means_, mixture_covariances=mixture.covariances_)
    recon = reconstruct(parameters, beats[:5])
    assert_allclose(recon, beats[:5], atol=2e-6)
    draws = draw(parameters, 12, 42)
    assert draws.shape == (12, 12, 400)
    assert_allclose(limb_errors(draws), 0, atol=1e-7)
    assert_array_equal(draws, draw(parameters, 12, 42))
    assert not np.array_equal(draws, draw(parameters, 12, 43))


def test_segment_evidence_rejects_linear_drift():
    scores = np.zeros(30)
    scores[5:7] = 0.9
    scores[10:22] = 0.85
    assert best_segment(scores, 8, 0, len(scores)) == (10, 21)
    flat = np.linspace(-0.2, 0.2, 30)
    assert segment_amplitude(flat, (10, 21)) < 1e-9
    flat[15] += 0.1
    assert_allclose(segment_amplitude(flat, (10, 21)), 0.1)
