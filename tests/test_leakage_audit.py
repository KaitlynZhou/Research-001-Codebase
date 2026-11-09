"""Leakage guards use artificial waveforms and never train on real test data."""
import numpy as np
import torch

from leakage_audit import near_duplicate_scan, overlap_counts, signal_hashes
from resnet_model import ECGDataset, ResNet1D, make_loaders, predict, run_epoch, train


def test_cross_split_identity_overlap():
    assert overlap_counts({'train': [1, 2], 'val': [3], 'test': [2, 4]}) == {
        'train/val': 0, 'train/test': 1, 'val/test': 0}


def test_waveform_hash_catches_copy_with_different_metadata():
    rng = np.random.default_rng(5)
    original = rng.normal(size=(2, 1, 5000)).astype(np.float32)
    hashes = signal_hashes(original)
    assert hashes[0] == signal_hashes(original[:1].copy())[0]
    assert hashes[0] != hashes[1]


def test_correlation_catches_scaled_offset_polarity_copy():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(1, 1, 5000)).astype(np.float32)
    result = near_duplicate_scan({'train': x, 'val': -3 * x + 10,
                                  'test': rng.normal(size=x.shape).astype(np.float32)})
    assert result['pairs']['train/val']['pairs_at_or_above_threshold'] == 1
    assert result['pairs']['train/test']['pairs_at_or_above_threshold'] == 0


def test_validation_freezes_parameters_and_batchnorm():
    torch.set_num_threads(1)
    model = ResNet1D(4)
    rng = np.random.default_rng(2)
    data = ECGDataset(rng.normal(size=(4, 5000)), [0, 1, 0, 1])
    before = {key: value.clone() for key, value in model.state_dict().items()}
    run_epoch(model, make_loaders({'val': data}, 4)['val'], 'cpu')
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    assert not model.training


def test_predictions_do_not_depend_on_labels():
    torch.set_num_threads(1)
    model = ResNet1D(4)
    x = np.random.default_rng(2).normal(size=(4, 5000))
    first = make_loaders({'test': ECGDataset(x, [0, 1, 0, 1])}, 4)['test']
    second = make_loaders({'test': ECGDataset(x, [1, 0, 1, 0])}, 4)['test']
    np.testing.assert_array_equal(predict(model, first, 'cpu')[1], predict(model, second, 'cpu')[1])


def test_weights_and_selection_never_read_test_or_reweight_validation(tmp_path, monkeypatch):
    import resnet_model
    observed = []
    training = ECGDataset(np.zeros((4, 5000)), [0, 0, 0, 1])
    validation = ECGDataset(np.zeros((4, 5000)), [0, 1, 1, 1])
    class Guard(dict):
        def __getitem__(self, key):
            assert key != 'test'
            return super().__getitem__(key)
    loaders = Guard(make_loaders({'train': training, 'val': validation}, 4))
    def capture(model, loader, device, optimizer=None, class_weights=None):
        observed.append(class_weights.tolist())
        return {'loss': .1, 'accuracy': .5, 'recall': .5, 'f1': .5}
    monkeypatch.setattr(resnet_model, 'run_epoch', capture)
    train(ResNet1D(4), loaders, 'cpu', tmp_path, epochs=1)
    np.testing.assert_allclose(observed, [[2 / 3, 2], [2 / 3, 2]])


def test_review_bundle_excludes_private_metadata_and_checkpoints(tmp_path):
    import json
    from leakage_audit import write_review_bundle
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'training_summary.json').write_text(json.dumps({'best_epoch': 8,
        'epochs_completed': 13, 'training_config': {'class_weights': [.5, 14.]}}))
    loss = [.5] * 13
    loss[7] = .1
    (run / 'history.json').write_text(json.dumps({'val_loss': loss, 'train_loss': loss}))
    metrics = {key: 1. for key in ('accuracy', 'precision', 'recall', 'f1', 'auroc',
                                   'pr_auc', 'average_precision', 'threshold')}
    metrics.update(confusion_matrix=[[613, 2], [0, 25]], metadata={'data_path': '/private/patient-data'})
    (run / 'metrics.json').write_text(json.dumps(metrics))
    (run / 'best_model.pt').write_bytes(b'private checkpoint')
    for name in ('training_curves.png', 'confusion_matrix.png', 'evaluation.png'):
        (run / name).write_bytes(b'test image')
    destination = tmp_path / 'review'
    write_review_bundle({'checkpoint_epoch': 8, 'checkpoint_sha256': 'a' * 64,
        'status': 'technical_checks_pass_with_historical_test_exposure_caveat'}, run, destination)
    names = {p.name for p in destination.iterdir()}
    assert names == {'README.md', 'audit.json', 'metrics.json', 'history.json', 'experiment.json',
                     'training_curves.png', 'confusion_matrix.png', 'evaluation.png'}
    sanitized = (destination / 'metrics.json').read_text()
    assert 'metadata' not in sanitized and '/private/' not in sanitized
    assert 'not an unconditional' in (destination / 'README.md').read_text()


def test_review_bundle_rejects_unresolved_findings(tmp_path):
    import pytest
    from leakage_audit import write_review_bundle
    with pytest.raises(ValueError, match='Resolve technical findings'):
        write_review_bundle({'checkpoint_epoch': 8, 'status': 'manual_near_duplicate_review_required'},
                            tmp_path, tmp_path / 'review')
