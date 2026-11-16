import numpy as np
import pandas as pd
import pytest
import torch
from non_diagnostic_leads import LEADS, LABELS, module, validate_metadata, metrics, dataset1_metadata


def test_six_lead_two_independent_outputs():
    assert LEADS == ['II', 'III', 'aVR', 'aVF', 'V3', 'V4']
    assert LABELS == ['CLBBB', 'ILBBB']
    model = module.ECGResNet(in_channels=6)
    logits = model(torch.randn(2, 6, 5000))
    assert logits.shape == (2, 2)
    torch.nn.BCEWithLogitsLoss()(logits, torch.tensor([[1., 1.], [0., 0.]])).backward()


def test_patient_leakage_rejected():
    rows = pd.DataFrame(dict(ecg_id=[1, 2], patient_id=[1, 1], strat_fold=[1, 10],
        filename_hr=['records500/a', 'records500/b'], CLBBB=[1, 0], ILBBB=[0, 1]))
    with pytest.raises(ValueError, match='Patient leakage'):
        validate_metadata(rows)


def test_team_policy_keeps_comorbid_lbbb_and_excludes_abnormal_normal(tmp_path):
    pd.DataFrame(dict(ecg_id=[1, 2, 3, 4], scp_codes=["{'NORM':100}",
        "{'NORM':100,'MI':100}", "{'CLBBB':50,'MI':100}", "{'ILBBB':0}"])).to_csv(tmp_path/'ptbxl_database.csv', index=False)
    pd.DataFrame(dict(diagnostic=[1, 1]), index=['NORM', 'MI']).to_csv(tmp_path/'scp_statements.csv')
    result = dataset1_metadata(tmp_path)
    assert result.ecg_id.tolist() == [1, 3, 4]
    assert result[LABELS].values.tolist() == [[0, 0], [1, 0], [0, 1]]


def test_per_label_metrics_fixed_threshold_and_confusion_order():
    result = metrics(np.array([[1, 0], [0, 1]]), np.array([[.5, .6], [.1, .2]]))
    assert result['CLBBB']['confusion_matrix'] == [[1, 0], [0, 1]]
    assert result['ILBBB']['confusion_matrix'] == [[0, 1], [1, 0]]


def test_training_checkpoint_and_evaluation_only(tmp_path):
    import argparse
    import json
    from non_diagnostic_leads import load, run
    directory = tmp_path/'dataset1'
    directory.mkdir()
    (directory/'contract.json').write_text(json.dumps(dict(dataset='dataset1',
        leads=LEADS, labels=LABELS, fs=500, samples=5000, normalization='none', source_column='filename_hr', waveform_directory='records500', units='mV', policy='synthetic software test')))
    rng = np.random.default_rng(1)
    for split, fold in [('train', 1), ('val', 9), ('test', 10)]:
        frame = pd.DataFrame(dict(ecg_id=[fold*10, fold*10+1],
            patient_id=[fold*10, fold*10+1], strat_fold=[fold, fold],
            filename_hr=[f'records500/{fold}a', f'records500/{fold}b'],
            CLBBB=[1, 0], ILBBB=[0, 1]))
        frame.to_csv(directory/f'{split}.csv', index=False)
        np.save(directory/f'X_{split}.npy', rng.normal(size=(2, 6, 5000)).astype(np.float32))
        np.save(directory/f'y_{split}.npy', frame[LABELS].to_numpy(dtype=np.float32))
    args = argparse.Namespace(dataset1=directory, dataset2=None, train='dataset1',
        output=tmp_path/'run', epochs=1, device='cpu', checkpoint=None)
    run(args)
    original = json.loads((args.output/'metrics.json').read_text())
    args.checkpoint = args.output/'best_model.pt'
    args.output = tmp_path/'evaluate'
    run(args)
    assert json.loads((args.output/'metrics.json').read_text()) == original
    # Corruption must fail before any training or evaluation.
    np.save(directory/'y_test.npy', np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match='Label alignment'):
        load(directory, 'dataset1')


@pytest.mark.parametrize('fs,samples', [(100, 1000), (100, 5000), (500, 1000)])
def test_prepare_rejects_wrong_frequency_or_sample_count(tmp_path, monkeypatch, fs, samples):
    import non_diagnostic_leads as pipeline
    frame = pd.DataFrame(dict(ecg_id=[1], patient_id=[1], strat_fold=[1],
        filename_hr=['records500/a'], CLBBB=[1], ILBBB=[0]))
    monkeypatch.setattr(pipeline, 'dataset1_metadata', lambda _: frame)
    monkeypatch.setattr(pipeline.wfdb, 'rdsamp', lambda _: (np.zeros((samples, 12)),
        dict(fs=fs, sig_name=['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6'], units=['mV']*12)))
    with pytest.raises(ValueError, match='Unexpected waveform contract'):
        pipeline.prepare(tmp_path, tmp_path/'export')


def test_prepare_500hz_selects_exact_channels_from_raw(tmp_path, monkeypatch):
    import json
    import non_diagnostic_leads as pipeline
    frame = pd.DataFrame(dict(ecg_id=[1, 2, 3], patient_id=[1, 2, 3], strat_fold=[1, 9, 10],
        filename_hr=['records500/a', 'records500/b', 'records500/c'], CLBBB=[1, 0, 1], ILBBB=[0, 1, 1]))
    monkeypatch.setattr(pipeline, 'dataset1_metadata', lambda _: frame)
    paths = []
    # Deliberately shuffled header order: selection must use names, not fixed positions.
    names = ['V6','V5','V4','V3','V2','V1','AVF','AVL','AVR','III','II','I']
    signal = np.broadcast_to(np.arange(12), (5000, 12)).astype(float)
    def read(path):
        paths.append(path)
        return signal, dict(fs=500, sig_name=names, units=['mV']*12)
    monkeypatch.setattr(pipeline.wfdb, 'rdsamp', read)
    for filename in ['ptbxl_database.csv', 'scp_statements.csv']:
        (tmp_path/filename).write_text('synthetic fixture')
    out = tmp_path/'export'
    pipeline.prepare(tmp_path, out)
    x = np.load(out/'X_train.npy')
    assert x.shape == (1, 6, 5000)
    np.testing.assert_array_equal(x[0, :, 0], [10, 9, 8, 6, 3, 2])
    assert all('records500/' in p for p in paths)
    contract = json.loads((out/'contract.json').read_text())
    assert (contract['fs'], contract['samples'], contract['source_column']) == (500, 5000, 'filename_hr')
    pipeline.load(out, 'dataset1')
    contract['fs'], contract['samples'] = 100, 1000
    (out/'contract.json').write_text(json.dumps(contract))
    with pytest.raises(ValueError, match='Wrong export contract'):
        pipeline.load(out, 'dataset1')


def test_low_resolution_manifest_rejected():
    frame = pd.DataFrame(dict(ecg_id=[1], patient_id=[1], strat_fold=[1],
        filename_hr=['records100/a'], CLBBB=[1], ILBBB=[0]))
    with pytest.raises(ValueError, match='records500'):
        validate_metadata(frame)
