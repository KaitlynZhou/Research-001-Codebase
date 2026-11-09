"""Artificial exports exercise Prisha's file contract, not clinical performance."""
import csv
import json

import numpy as np
import pytest
import torch

from ptbxl_data import load_v6_exports, strict_label, verify_waveform
from resnet_model import ECGDataset, ResNet1D, make_loaders, train, load_checkpoint, evaluate


@pytest.fixture
def exports(tmp_path):
    config = {'lead': 'V6', 'fs': 500, 'samples': 5000, 'units': 'mV',
              'X_dtype': 'float32', 'y_dtype': 'int64',
              'normalization': 'none; fit any later normalization on training only',
              'manifest_sha256': '0' * 64}
    (tmp_path / 'preprocessing.json').write_text(json.dumps(config))
    for index, split in enumerate(('train', 'val', 'test')):
        np.save(tmp_path / f'X_{split}.npy', np.zeros((4, 1, 5000), dtype=np.float32))
        np.save(tmp_path / f'y_{split}.npy', np.array([0, 1, 0, 1], dtype=np.int64))
        rows = [{'ecg_id': index * 10 + j, 'patient_id': index * 10 + j,
                 'strat_fold': [1, 9, 10][index], 'split': split, 'label': j % 2,
                 'scp_codes': str({'CLBBB' if j % 2 else 'NORM': 100, 'SR': 0}),
                 'filename_hr': f'records500/00000/{index * 10 + j:05d}_hr'} for j in range(4)]
        write_rows(tmp_path / f'{split}_manifest.csv', rows)
    return tmp_path


def write_rows(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_export_training_round_trip(exports, tmp_path):
    torch.set_num_threads(1)
    arrays, provenance = load_v6_exports(exports)
    assert provenance['patient_overlap_between_splits'] == 0
    assert not provenance['all_rows_verified_against_raw_v6']
    assert provenance['split_counts']['train']['CLBBB'] == 2
    loaders = make_loaders({key: ECGDataset(*value) for key, value in arrays.items()}, 4)
    output = tmp_path / 'output'
    train(ResNet1D(4), loaders, 'cpu', output, epochs=1,
          metadata={'synthetic': True, 'data_provenance': provenance})
    model, checkpoint = load_checkpoint(output / 'best_model.pt', 'cpu')
    assert checkpoint['metadata']['data_provenance'] == provenance
    assert evaluate(model, loaders['test'], 'cpu', output, {'synthetic': True})['accuracy'] >= 0


@pytest.mark.parametrize('field,value,message', [
    ('patient_id', 0, 'Patient leakage'),
    ('ecg_id', 0, 'Duplicate recording'),
    ('filename_hr', 'records500/00000/00000_hr', 'Duplicate recording'),
    ('strat_fold', 10, 'split disagrees'),
    ('split', 'train', 'split disagrees'),
    ('label', 1, 'labels disagree'),
    ('scp_codes', "{'CLBBB': 100, 'ILBBB': 0}", 'strict-v1'),
    ('patient_id', '', 'Invalid integer'),
    ('filename_hr', '../escape', 'waveform path'),
])
def test_manifest_rejection(exports, field, value, message):
    path = exports / 'val_manifest.csv'
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    rows[0][field] = value
    write_rows(path, rows)
    with pytest.raises(ValueError, match=message):
        load_v6_exports(exports)


@pytest.mark.parametrize('field,value', [('lead', 'V1'), ('fs', 100), ('samples', 1000),
                                       ('units', 'uV'), ('normalization', 'global z-score')])
def test_wrong_preprocessing(exports, field, value):
    path = exports / 'preprocessing.json'
    config = json.loads(path.read_text())
    config[field] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='Unexpected preprocessing'):
        load_v6_exports(exports)


def test_label_row_alignment(exports):
    np.save(exports / 'y_val.npy', np.array([1, 0, 0, 1], dtype=np.int64))
    with pytest.raises(ValueError, match='labels disagree'):
        load_v6_exports(exports)


def test_patient_cannot_move_between_training_folds(exports):
    path = exports / 'train_manifest.csv'
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    rows[1]['patient_id'], rows[1]['strat_fold'] = rows[0]['patient_id'], 2
    write_rows(path, rows)
    with pytest.raises(ValueError, match='Patient leakage'):
        load_v6_exports(exports)


@pytest.mark.parametrize('codes', ["{'CLBBB': 80}", "{'CLBBB': 100, 'NORM': 100}",
                                  "{'NORM': 100, 'LVH': 0}", "{'ILBBB': 100}"])
def test_strict_policy(codes):
    with pytest.raises(ValueError, match='strict-v1'):
        strict_label(codes)


def test_raw_v6_verification(tmp_path):
    import wfdb
    x = np.linspace(-1, 1, 5000)
    for name, rate, lead, unit in [('valid', 500, 'V6', 'mV'),
                                    ('rate', 100, 'V6', 'mV'),
                                    ('lead', 500, 'V5', 'mV'),
                                    ('unit', 500, 'V6', 'uV')]:
        wfdb.wrsamp(name, fs=rate, units=['mV', unit], sig_name=['I', lead],
                    p_signal=np.column_stack([np.zeros(5000), x]), fmt=['16', '16'],
                    write_dir=str(tmp_path))
    saved = wfdb.rdrecord(str(tmp_path / 'valid')).p_signal[:, 1].astype(np.float32)
    verify_waveform(tmp_path / 'valid', saved)
    for name in ('rate', 'lead', 'unit'):
        with pytest.raises(ValueError):
            verify_waveform(tmp_path / name, saved)
    with pytest.raises(ValueError, match='does not match'):
        verify_waveform(tmp_path / 'valid', np.zeros(5000))


def test_missing_export_does_not_fall_back_to_synthetic(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_v6_exports(tmp_path)


def test_cli_checkpoint_and_changed_exports(exports, tmp_path, monkeypatch):
    import sys
    from resnet_model import main
    torch.set_num_threads(1)
    output = tmp_path / 'cli-output'
    arguments = ['resnet_model.py', '--data', str(exports), '--device', 'cpu',
                 '--output', str(output), '--epochs', '1']
    monkeypatch.setattr(sys, 'argv', arguments)
    main()
    metadata = json.loads((output / 'run_metadata.json').read_text())
    assert metadata['synthetic'] is False
    arguments += ['--checkpoint', str(output / 'best_model.pt')]
    monkeypatch.setattr(sys, 'argv', arguments)
    main()
    signals = np.load(exports / 'X_train.npy')
    signals[0, 0, 0] = 1
    np.save(exports / 'X_train.npy', signals)
    with pytest.raises(ValueError, match='exports differ'):
        main()


def test_cli_requires_explicit_synthetic_mode(tmp_path, monkeypatch):
    import sys
    from resnet_model import main
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['resnet_model.py'])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_adapted_prisha_extractor_and_unchanged_cohort_logic(tmp_path, monkeypatch):
    import importlib
    import subprocess
    import wfdb
    from prepare_ptbxl import export_pipeline, PREPROCESSING_COMMIT
    source = export_pipeline(tmp_path / 'pipeline')
    for name in ('inspect_metadata.py', 'process_metadata.py'):
        original = subprocess.check_output([
            'git', 'show', f'{PREPROCESSING_COMMIT}:src/ecg_lbbb/{name}'])
        assert (source / 'ecg_lbbb' / name).read_bytes() == original
    monkeypatch.syspath_prepend(str(source))
    extractor = importlib.import_module('ecg_lbbb.extract_v6')
    x = np.linspace(-1, 1, 5000)
    for name, rate in [('original500', 500), ('reject100', 100)]:
        wfdb.wrsamp(name, fs=rate, units=['mV', 'mV'], sig_name=['V6', 'I'],
                    p_signal=np.column_stack([x, np.zeros(5000)]), fmt=['16', '16'],
                    write_dir=str(tmp_path))
    extracted = extractor.load_v6(tmp_path / 'original500')
    assert extracted.shape == (1, 5000)
    verify_waveform(tmp_path / 'original500', extracted)
    with pytest.raises(ValueError, match='500 Hz'):
        extractor.load_v6(tmp_path / 'reject100')


def test_reject_low_resolution_path(exports):
    path = exports / 'val_manifest.csv'
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    rows[0]['filename_hr'] = 'records100/00000/00010_lr'
    write_rows(path, rows)
    with pytest.raises(ValueError, match='records500'):
        load_v6_exports(exports)
