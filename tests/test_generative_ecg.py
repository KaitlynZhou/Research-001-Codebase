import os
from pathlib import Path

import numpy as np
import pytest

from generative_ecg.beats import PRE, SHAPE, extract, read_record
from generative_ecg.evaluate import limb_errors, morphology


def test_detect_align_and_reject_artifact():
    samples = np.arange(5000)
    anchors = np.arange(350, 4700, 500)
    wave = np.zeros((12, len(samples)))
    amplitude = np.linspace(-0.6, 1.0, 12)
    for anchor in anchors:
        wave += amplitude[:, None] * np.exp(-0.5 * ((samples - anchor) / 9)**2)
    wave[0, anchors[3] + 125:anchors[3] + 130] = 15
    median, detected, beats = extract(wave)
    assert median.shape == SHAPE
    assert len(detected) == len(anchors) - 1
    np.testing.assert_allclose(detected, np.delete(anchors, 3), atol=2)
    assert np.argmax(median[-1]) == PRE
    assert np.argmin(median[0]) == PRE
    np.testing.assert_allclose(median, np.median(beats, axis=0), atol=1e-7)


@pytest.mark.parametrize("wave, reason", [
    (np.zeros((12, 5000)), "weak_qrs"),
    (np.full((12, 5000), np.nan), "invalid_waveform"),
    (np.zeros((6, 5000)), "invalid_waveform"),
])
def test_reject_invalid_recording(wave, reason):
    with pytest.raises(ValueError, match=reason):
        extract(wave)


def test_read_uses_all_leads_at_native_rate(monkeypatch):
    from generative_ecg.beats import LEADS

    reversed_signal = np.tile(np.arange(12)[::-1], (5000, 1))
    fields = dict(fs=500, units=["mV"] * 12, sig_name=list(reversed(LEADS)))
    monkeypatch.setattr("wfdb.rdsamp", lambda path: (reversed_signal, fields))
    wave = read_record(Path("."), "records500/test")
    assert wave.shape == (12, 5000)
    np.testing.assert_array_equal(wave[:, 0], np.arange(12))
    with pytest.raises(ValueError, match="records500"):
        read_record(Path("."), "records100/test")
    fields["fs"] = 100
    with pytest.raises(ValueError, match="waveform contract"):
        read_record(Path("."), "records500/test")


def test_evaluation_respects_lead_identities_and_missing_waves():
    values = np.zeros((2, *SHAPE))
    values[:, 0] = 2
    values[:, 1] = 3
    values[:, 2] = 1
    values[:, 3] = -2.5
    values[:, 4] = 0.5
    values[:, 5] = 2
    np.testing.assert_array_equal(limb_errors(values), 0)
    assert np.isnan(morphology(np.zeros(SHAPE))).all()


@pytest.mark.skipif(not os.environ.get("ECG_SCD_ROOT"), reason="Requires upstream source")
def test_upstream_network_shapes():
    import jax
    import jax.numpy as jnp

    from generative_ecg.model import LATENT, network

    original, _, encoder, decoder = network(Path(os.environ["ECG_SCD_ROOT"]))
    key = jax.random.PRNGKey(42)
    wave = jnp.zeros(SHAPE)
    mu, variance = encoder.apply(encoder.init(key, wave), wave)
    assert mu.shape == variance.shape == (LATENT,)
    assert np.all(variance > 0)
    z = original.gaussian_sample(key, mu, variance)
    assert decoder.apply(decoder.init(key, z), z).shape == (np.prod(SHAPE),)
