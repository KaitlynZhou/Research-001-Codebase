"""Independently verify downloaded coverage, unchanged policy, saved arrays and splits."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .inspect_metadata import parse_codes
from .process_metadata import classify, verify_splits
from .extract_v6 import load_v6


def main():
    root = Path('data/raw/ptb-xl')
    output = Path('data/processed/v6')
    original = json.loads(Path('results/pre_download_hashes.json').read_text())
    for path, expected in original.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    metadata = pd.read_csv(root / 'ptbxl_database.csv')
    selected = pd.read_csv('data/processed/selected_recordings.csv')
    verify_splits(metadata)
    verify_splits(selected)
    decisions = metadata.scp_codes.map(lambda s: classify(parse_codes(s))[0])
    expected_ids = set(metadata.loc[decisions.notna(), 'ecg_id'])
    assert expected_ids == set(selected.ecg_id)
    labels = dict(zip(metadata.ecg_id, decisions))
    assert all(labels[row.ecg_id] == row.label for row in selected.itertuples())
    missing = [str(root / f'{name}{suffix}') for name in metadata.filename_lr
               for suffix in ['.hea', '.dat'] if not (root / f'{name}{suffix}').is_file()]
    assert not missing, f'Missing metadata waveform files: {len(missing)}'
    download = json.loads((root / 'records100-download-report.json').read_text())
    assert not download['failures'] and download['sha256_verified'] == 2 * len(metadata)
    sets, results = {}, {}
    for split in ['train', 'val', 'test']:
        rows = selected.loc[selected.split == split].reset_index(drop=True)
        saved_rows = pd.read_csv(output / f'{split}_manifest.csv')
        pd.testing.assert_frame_equal(rows, saved_rows)
        X = np.load(output / f'X_{split}.npy', allow_pickle=False)
        y = np.load(output / f'y_{split}.npy', allow_pickle=False)
        assert X.shape == (len(rows), 1, 1000) and X.dtype == np.float32
        assert y.shape == (len(rows),) and y.dtype == np.int64
        assert np.isfinite(X).all()
        np.testing.assert_array_equal(y, rows.label.to_numpy())
        # Check every saved row against its source V6 waveform, not just one example.
        for index, path in enumerate(rows.filename_lr):
            np.testing.assert_array_equal(X[index], load_v6(root / path))
        sets[split] = set(rows.patient_id)
        results[split] = {'X_shape': list(X.shape), 'y_shape': list(y.shape),
                          'normal': int((y == 0).sum()), 'CLBBB': int((y == 1).sum()),
                          'patients': len(sets[split]), 'nonfinite_samples': 0,
                          'all_rows_match_source_v6': True}
    assert not (sets['train'] & sets['val'] or sets['train'] & sets['test'] or sets['val'] & sets['test'])
    report = {'version': '1.0.3', 'metadata_recordings_with_both_files': len(metadata),
              'official_sha256_verified_files': download['sha256_verified'],
              'original_metadata_and_selection_unchanged': True, 'patient_overlap': 0,
              'conservative_labels_verified': True, 'splits': results}
    Path('results/v6_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
