"""Regression checks: median-beat reader, PTB-XL+ path layout, and architecture parity with the six-lead model."""
import tempfile
from pathlib import Path
import numpy as np
import torch
from common import read_median, median_relpath, six_lead_model, LEADS
from model import ECGResNet

NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def write_median(path, microvolts):
    """Write a record in the PTB-XL+ layout: directory in record name, int32, each lead's range
    mapped onto the full int32 range via its own gain(baseline)/mV, WFDB checksum."""
    lo, hi = microvolts.min(axis=0), microvolts.max(axis=0)
    gains = (2**32 - 2) / (hi - lo)
    baselines = np.round(-2**31 + 1 - lo * gains).astype(np.int64)
    digital = np.round(microvolts * gains + baselines).astype(np.int64)
    assert digital.min() >= -2**31 and digital.max() < 2**31
    digital = digital.astype('<i4')
    path.parent.mkdir(parents=True, exist_ok=True)
    digital.tofile(path.with_suffix('.dat'))
    lines = [f'ge_median_beats_wfdb/{path.name} 12 500 600']
    for i, n in enumerate(NAMES):
        col = digital[:, i].astype(np.int64)
        lines.append(f'ge_median_beats_wfdb/{path.name}.dat 32 {float(gains[i])!r}({int(baselines[i])})/mV 32 0 {col[0]} {int(col.sum()) % 65536} 0 {n}')
    path.with_suffix('.hea').write_text('\n'.join(lines) + '\n')
    return digital


def main():
    assert median_relpath(1) == 'median_beats/12sl/00000/00001_medians'
    assert median_relpath(21837) == 'median_beats/12sl/21000/21837_medians'
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / '00001_medians'
        uv = np.random.default_rng(0).integers(-900, 900, (600, 12)).astype(float)
        write_median(p, uv)
        beat = read_median(p)
        for i, n in enumerate(NAMES):
            np.testing.assert_allclose(beat[n.upper()], uv[:, i] / 1000, atol=1e-6)  # microvolts -> mV
        raw = np.fromfile(p.with_suffix('.dat'), '<i4'); raw[100] += 7; raw.tofile(p.with_suffix('.dat'))
        try: read_median(p)
        except ValueError: pass
        else: raise AssertionError('Corrupted median beat was not rejected')
    # Only the stem's input-channel count differs from the six-lead model.
    six, one = six_lead_model.ECGResNet(), ECGResNet(1)
    s = six.state_dict(); s['stem.0.weight'] = s['stem.0.weight'][:, [2], :]  # V1 is channel 2 of I,aVL,V1,V2,V5,V6
    one.load_state_dict(s); six.eval(); one.eval()
    x = torch.randn(3, 1, 5000); full = torch.zeros(3, 6, 5000); full[:, 2:3] = x
    with torch.no_grad(): torch.testing.assert_close(six(full), one(x))
    assert one(torch.randn(2, 1, 600)).shape == (2, 2)  # median-beat length works too
    assert LEADS == ['V1', 'V2', 'V3', 'V4']
    print('PASS: median-beat reader (units, checksum), PTB-XL+ paths, architecture parity')


if __name__ == '__main__':
    main()
