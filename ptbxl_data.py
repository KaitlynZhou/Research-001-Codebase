"""Read Prisha's V6 exports without modifying or rerunning her pipeline."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SPLITS = ('train', 'val', 'test')
POLICY = 'strict-v1: target=100; only target and optional SR; all other code presence excludes'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def integer(value, field):
    try:
        number = float(value)
        if not np.isfinite(number) or number != int(number):
            raise ValueError
        return int(number)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f'Invalid integer {field}: {value}') from error


def strict_label(text):
    """Validate the same strict-v1 policy as Prisha's process_metadata.classify."""
    try:
        codes = ast.literal_eval(text)
    except (ValueError, SyntaxError) as error:
        raise ValueError('Invalid scp_codes dictionary') from error
    if not isinstance(codes, dict):
        raise ValueError('Invalid scp_codes dictionary')
    if not codes or any(not isinstance(code, str) or isinstance(value, bool)
                        or not isinstance(value, (int, float)) or not math.isfinite(value)
                        or not 0 <= value <= 100 for code, value in codes.items()):
        raise ValueError('Invalid SCP code or confidence')
    targets = set(codes) & {'NORM', 'CLBBB'}
    if len(targets) != 1:
        raise ValueError('Manifest violates strict-v1 target policy')
    target = next(iter(targets))
    if codes[target] != 100 or set(codes) - {target, 'SR'}:
        raise ValueError('Manifest violates strict-v1 target policy')
    return int(target == 'CLBBB')


def waveform_path(root, filename):
    relative = Path(filename)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts or relative.parts[0] != 'records500':
        raise ValueError(f'Invalid records500 waveform path: {filename}')
    path = Path(root) / relative
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Waveform path escapes data root')
    return path


def verify_waveform(path, expected):
    """Optional independent all-row comparison with local source V6 waveforms."""
    try:
        import wfdb
    except ImportError as error:
        raise ImportError('Install requirements.txt to verify raw WFDB recordings') from error
    record = wfdb.rdrecord(str(path))
    names = [name.upper() for name in record.sig_name]
    if record.fs != 500 or record.sig_len != 5000 or names.count('V6') != 1:
        raise ValueError(f'{path}: expected 500 Hz, 5000 samples and one V6 lead')
    channel = names.index('V6')
    if record.units[channel].lower() != 'mv':
        raise ValueError(f'{path}: V6 must be in mV')
    signal = record.p_signal[:, channel].astype(np.float32)
    if not np.isfinite(signal).all() or not np.array_equal(signal, expected.reshape(-1)):
        raise ValueError(f'{path}: saved signal does not match raw V6')


def load_v6_exports(directory, raw_root=None):
    """Return validated arrays plus provenance. No download, resplit or normalization."""
    directory = Path(directory)
    config = json.loads((directory / 'preprocessing.json').read_text())
    expected = {'lead': 'V6', 'fs': 500, 'samples': 5000, 'units': 'mV',
                'X_dtype': 'float32', 'y_dtype': 'int64',
                'normalization': 'none; fit any later normalization on training only'}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Unexpected preprocessing {key}: {config.get(key)!r}')
    if not isinstance(config.get('manifest_sha256'), str) or len(config['manifest_sha256']) != 64 or any(
            char not in '0123456789abcdef' for char in config['manifest_sha256']):
        raise ValueError('Missing or invalid selection manifest SHA-256')
    arrays, fingerprints, counts = {}, {}, {}
    patients, recordings, paths = {}, set(), set()
    for split in SPLITS:
        manifest = directory / f'{split}_manifest.csv'
        with manifest.open(newline='') as stream:
            reader = csv.DictReader(stream)
            required = {'ecg_id', 'patient_id', 'scp_codes', 'strat_fold', 'filename_hr', 'label', 'split'}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f'{manifest}: missing required manifest columns')
            rows = list(reader)
        x_path, y_path = directory / f'X_{split}.npy', directory / f'y_{split}.npy'
        x, y = np.load(x_path, allow_pickle=False), np.load(y_path, allow_pickle=False)
        if x.dtype != np.float32 or x.shape != (len(rows), 1, 5000) or not len(rows) or not np.isfinite(x).all():
            raise ValueError(f'{split}: expected finite float32 (N, 1, 5000) signals')
        if y.dtype != np.int64 or y.shape != (len(rows),) or set(y.tolist()) != {0, 1}:
            raise ValueError(f'{split}: expected int64 labels with both classes')
        for index, row in enumerate(rows):
            patient = integer(row['patient_id'], 'patient_id')
            ecg = integer(row['ecg_id'], 'ecg_id')
            fold = integer(row['strat_fold'], 'strat_fold')
            label = integer(row['label'], 'label')
            if fold not in range(1, 11):
                raise ValueError('Invalid strat_fold')
            expected_split = 'train' if fold <= 8 else 'val' if fold == 9 else 'test'
            if row['split'] != split or expected_split != split:
                raise ValueError('Manifest split disagrees with official folds')
            if label != y[index] or label != strict_label(row['scp_codes']):
                raise ValueError('Manifest labels disagree with arrays or strict-v1 policy')
            if patient in patients and patients[patient] != fold:
                raise ValueError('Patient leakage between folds or splits')
            patients[patient] = fold
            if ecg in recordings or row['filename_hr'] in paths:
                raise ValueError('Duplicate recording ID or waveform path')
            recordings.add(ecg)
            paths.add(row['filename_hr'])
            source_path = waveform_path(raw_root or directory, row['filename_hr'])
            if raw_root is not None:
                verify_waveform(source_path, x[index])
        arrays[split] = (x, y)
        counts[split] = {'recordings': len(y), 'normal': int((y == 0).sum()),
                         'CLBBB': int((y == 1).sum()),
                         'patients': len({integer(r['patient_id'], 'patient_id') for r in rows})}
        for path in (manifest, x_path, y_path):
            fingerprints[path.name] = sha256(path)
    fingerprints['preprocessing.json'] = sha256(directory / 'preprocessing.json')
    return arrays, {'format': 'prisha-v6-directory', 'label_policy': POLICY,
                    'class_names': ['Normal', 'CLBBB'], 'preprocessing': config,
                    'file_sha256': fingerprints, 'split_counts': counts,
                    'patient_overlap_between_splits': 0,
                    'all_rows_verified_against_raw_v6': raw_root is not None}
