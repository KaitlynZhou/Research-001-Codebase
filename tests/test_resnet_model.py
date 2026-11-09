import numpy as np
import pytest
import torch

from resnet_model import (ECGDataset, ResNet1D, evaluate, load_checkpoint,
                          load_processed_data, make_loaders, predict, set_seed,
                          synthetic_data, train)


@pytest.fixture(autouse=True)
def reproducibility():
    set_seed(7)
    torch.set_num_threads(1)


@pytest.mark.parametrize('shape', [(4, 5000), (4, 1, 5000)])
def test_shapes_forward_backward(shape):
    dataset = ECGDataset(np.zeros(shape), [0, 1, 0, 1])
    assert dataset.x.shape == (4, 1, 5000)
    model = ResNet1D(base_channels=4)
    logits = model(dataset.x)
    assert logits.shape == (4, 2)
    loss = torch.nn.CrossEntropyLoss()(logits, dataset.y)
    loss.backward()
    assert model.classifier.weight.grad is not None
    assert torch.isfinite(model.classifier.weight.grad).all()


def test_complete_pipeline(tmp_path):
    loaders = make_loaders(synthetic_data(7, (12, 6, 6)), batch_size=4)
    model = ResNet1D(base_channels=4)
    before = model.classifier.weight.detach().clone()
    history = train(model, loaders, torch.device('cpu'), tmp_path, epochs=2,
                    metadata={'synthetic': True})
    assert not torch.equal(before, model.classifier.weight)
    assert len(history['val_loss']) == 2
    restored, checkpoint = load_checkpoint(tmp_path / 'best_model.pt', 'cpu')
    assert checkpoint['val_loss'] == min(history['val_loss'])
    clone = ResNet1D(base_channels=4).eval()
    clone.load_state_dict(checkpoint['model_state_dict'])
    x, _ = next(iter(loaders['val']))
    with torch.no_grad():
        torch.testing.assert_close(restored(x), clone(x))
    y, p = predict(restored, loaders['test'], torch.device('cpu'))
    assert y.shape == p.shape == (6,)
    assert ((p >= 0) & (p <= 1)).all()
    metrics = evaluate(restored, loaders['test'], 'cpu', tmp_path, {'synthetic': True})
    for key in ('accuracy', 'precision', 'recall', 'f1', 'auroc', 'pr_auc'):
        assert 0 <= metrics[key] <= 1
    assert np.asarray(metrics['confusion_matrix']).sum() == 6
    for name in ('best_model.pt', 'history.json', 'training_curves.png',
                 'metrics.json', 'evaluation.png', 'predictions.npz'):
        assert (tmp_path / name).stat().st_size > 0


def write_real_fixture(path, overlap=False, missing=False):
    arrays = {}
    for index, split in enumerate(('train', 'val', 'test')):
        arrays[f'X_{split}'] = np.zeros((4, 5000))
        arrays[f'y_{split}'] = np.array([0, 1, 0, 1])
        if not missing:
            arrays[f'patient_ids_{split}'] = np.arange(4) + (0 if overlap else index * 4)
    np.savez(path, **arrays)


def test_patient_separation(tmp_path):
    path = tmp_path / 'real.npz'
    write_real_fixture(path)
    assert len(load_processed_data(path)['train']) == 4
    write_real_fixture(path, overlap=True)
    with pytest.raises(ValueError, match='Patient overlap'):
        load_processed_data(path)
    write_real_fixture(path, missing=True)
    with pytest.raises(ValueError, match='mandatory'):
        load_processed_data(path)


@pytest.mark.parametrize('x,y', [
    (np.zeros((2, 12, 5000)), [0, 1]),
    (np.zeros((2, 999)), [0, 1]),
    (np.zeros((2, 5000)), [0, 2]),
    (np.zeros((2, 5000)), [[0], [1]]),
    (np.full((2, 5000), np.nan), [0, 1]),
])
def test_invalid_data(x, y):
    with pytest.raises(ValueError):
        ECGDataset(x, y)


def test_single_class_metrics(tmp_path):
    loader = make_loaders({'test': ECGDataset(np.zeros((2, 5000)), [0, 0])})['test']
    metrics = evaluate(ResNet1D(4), loader, 'cpu', tmp_path)
    assert metrics['auroc'] is None
    assert metrics['pr_auc'] is None


def test_seed_repeats_initialization():
    set_seed(11)
    first = ResNet1D(4).classifier.weight.detach().clone()
    set_seed(11)
    torch.testing.assert_close(first, ResNet1D(4).classifier.weight)


def test_training_does_not_access_test_split(tmp_path):
    class GuardedLoaders(dict):
        def __getitem__(self, key):
            if key == 'test':
                raise AssertionError('Training must not access test data')
            return super().__getitem__(key)

    loaders = GuardedLoaders(make_loaders(synthetic_data(7, (4, 4, 4)), 4))
    train(ResNet1D(4), loaders, 'cpu', tmp_path, epochs=1)


def test_reject_100hz_inputs_and_checkpoint(tmp_path):
    with pytest.raises(ValueError, match='5000'):
        ECGDataset(np.zeros((2, 1000)), [0, 1])
    with pytest.raises(ValueError, match='5000'):
        ResNet1D(4)(torch.zeros(2, 1, 1000))
    old_checkpoint = tmp_path / 'old_100hz.pt'
    torch.save({'input_samples': 1000, 'sampling_rate_hz': 100}, old_checkpoint)
    with pytest.raises(ValueError, match='500 Hz'):
        load_checkpoint(old_checkpoint, 'cpu')
