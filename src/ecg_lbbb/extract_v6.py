"""Extract local 100 Hz V6 waveforms into ResNet-ready NumPy arrays."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
from .process_metadata import verify_splits


def load_v6(path):
    record = wfdb.rdrecord(str(path))
    names = [s.upper() for s in record.sig_name]
    if record.fs != 100 or record.sig_len != 1000 or names.count('V6') != 1:
        raise ValueError(f'{path}: expected 100 Hz, 1000 samples, one V6 lead')
    channel = names.index('V6')
    if record.units[channel].lower() != 'mv':
        raise ValueError(f'{path}: expected mV units')
    signal = record.p_signal[:, channel].astype(np.float32)
    if signal.shape != (1000,) or not np.isfinite(signal).all():
        raise ValueError(f'{path}: invalid V6 samples')
    return signal[None, :]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('data/raw/ptb-xl'))
    p.add_argument('--manifest', type=Path, default=Path('data/processed/selected_recordings.csv'))
    p.add_argument('--output', type=Path, default=Path('data/processed/v6'))
    args = p.parse_args()
    df = pd.read_csv(args.manifest)
    verify_splits(df)
    expected = df.strat_fold.map(lambda f: 'train' if f <= 8 else 'val' if f == 9 else 'test')
    if not df.split.equals(expected) or not df.label.isin([0, 1]).all():
        raise ValueError('Invalid split or labels in manifest')
    paths = []
    for value in df.filename_lr:
        relative = Path(value)
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'records100':
            raise ValueError(f'Unexpected 100 Hz waveform path: {value}')
        paths.append(args.data_root / relative)
    missing = [str(p) for p in paths if not p.with_suffix('.hea').is_file() or not p.with_suffix('.dat').is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} waveform pairs unavailable; first: {missing[0]}. No download attempted.')
    outputs = {}
    for split in ['train', 'val', 'test']:
        rows = df.loc[df.split == split]
        if set(rows.label) != {0, 1}:
            raise ValueError(f'{split} missing a class')
        X = np.stack([load_v6(args.data_root / value) for value in rows.filename_lr])
        outputs[split] = (rows, X, rows.label.to_numpy(dtype=np.int64))
    args.output.mkdir(parents=True, exist_ok=True)
    for split, (rows, X, y) in outputs.items():
        np.save(args.output / f'X_{split}.npy', X, allow_pickle=False)
        np.save(args.output / f'y_{split}.npy', y, allow_pickle=False)
        rows.to_csv(args.output / f'{split}_manifest.csv', index=False)
    (args.output / 'preprocessing.json').write_text(json.dumps({
        'lead': 'V6', 'fs': 100, 'samples': 1000, 'units': 'mV',
        'X_dtype': 'float32', 'y_dtype': 'int64',
        'normalization': 'none; fit any later normalization on training only',
        'manifest_sha256': __import__('hashlib').sha256(args.manifest.read_bytes()).hexdigest()
    }, indent=2))


if __name__ == '__main__':
    main()
