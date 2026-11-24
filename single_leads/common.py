"""Shared definitions for the single-lead V1-V4 experiment (full 10-s ECG vs PTB-XL+ median beat)."""
import importlib.util, re
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SIX_LEAD = HERE.parent / 'diagnostic_leads'


def _load(name, path):
    """Load a six-lead module by path (both folders have prepare.py / model.py, so no sys.path import)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


# Dataset 1 / Dataset 2 labels, folds and leakage checks, unchanged from the six-lead branch.
cohorts = _load('six_lead_prepare', SIX_LEAD / 'prepare.py').cohorts
six_lead_model = _load('six_lead_model', SIX_LEAD / 'model.py')

LEADS = ['V1', 'V2', 'V3', 'V4']
OUTCOMES = ['CLBBB', 'ILBBB']
INPUTS = {'full': dict(fs=500, samples=5000, description='full 10-second ECG, native 500 Hz'),
          'median': dict(fs=500, samples=600, description='PTB-XL+ GE 12SL median beat, 1.2 s at 500 Hz')}
SPLITS = ['train', 'val', 'dataset2_test']
MEDIAN_SUBDIR = 'median_beats/12sl'
SEED = 42


def median_relpath(ecg_id):
    """PTB-XL+ layout: median_beats/12sl/<thousands folder>/<ecg_id>_medians.{hea,dat}."""
    return f'{MEDIAN_SUBDIR}/{ecg_id // 1000 * 1000:05d}/{ecg_id:05d}_medians'


_GAIN = re.compile(r'^([-+\d.eE]+)\(([-+\d]+)\)/(\w+)$')


def read_median(path):
    """Read one PTB-XL+ 12SL median-beat record and return {lead: samples in mV}.

    The released headers put a directory in the record name, which the wfdb package rejects,
    so the header is parsed here. Every channel's WFDB checksum and initial value is verified.

    Encoding, as found in the files: each lead's gain and baseline map its own range onto the
    full int32 range, and physical = (stored - baseline) / gain computed in 64-bit arithmetic
    (32-bit arithmetic wraps and corrupts the waveform). Decoded values are exact integers,
    which is asserted. The headers say mV, but those integers are microvolts (a normal V2 QRS
    spans ~1.6 mV, stored as ~1600), so values are divided by 1000.
    """
    path = Path(path)
    lines = [l for l in path.with_suffix('.hea').read_text().splitlines() if l and not l.startswith('#')]
    _, nsig, fs, n = lines[0].split()[:4]
    nsig, fs, n = int(nsig), int(float(fs)), int(n)
    if (nsig, fs, n) != (12, 500, 600):
        raise ValueError(f'Unexpected median-beat header in {path}')
    raw = np.fromfile(path.with_suffix('.dat'), dtype='<i4')
    if raw.size != nsig * n:
        raise ValueError(f'Truncated median-beat data in {path}')
    raw = raw.reshape(n, nsig)
    out = {}
    for i, line in enumerate(lines[1:1 + nsig]):
        f = line.split()
        if f[1] != '32':
            raise ValueError(f'Unexpected storage format in {path}')
        m = _GAIN.match(f[2])
        gain, baseline = float(m[1]), int(m[2])
        initial, checksum = int(f[5]), int(f[6])
        column = raw[:, i].astype(np.int64)
        if column[0] != initial or (int(column.sum()) - checksum) % 65536:
            raise ValueError(f'Checksum mismatch for {f[-1]} in {path}')
        microvolts = (column - baseline) / gain
        if np.abs(microvolts - np.round(microvolts)).max() > 1e-3:
            raise ValueError(f'{f[-1]} in {path} does not decode to integer microvolts')
        out[f[-1].upper()] = (np.round(microvolts) / 1000.0).astype(np.float32)
    return out
