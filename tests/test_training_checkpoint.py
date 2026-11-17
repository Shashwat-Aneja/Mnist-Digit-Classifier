import importlib
import torch
import pytest
import train
from model import load_model


def test_training_checkpoint_round_trips_through_shared_loader(tmp_path):
    data = [(torch.zeros((2, 1, 28, 28)), torch.tensor([0, 1]))]
    output = tmp_path / "weights.pth"
    trained = train.train_model(output, epochs=1, train_loader=data)
    loaded = load_model(output)
    assert set(trained.state_dict()) == set(loaded.state_dict())
    for key in trained.state_dict():
        assert torch.equal(trained.state_dict()[key], loaded.state_dict()[key])


def test_importing_training_module_does_not_download_or_train(monkeypatch):
    monkeypatch.setattr(train.datasets, "MNIST", lambda **kwargs: pytest.fail("import downloaded data"))
    importlib.reload(train)


def test_empty_training_data_does_not_save_checkpoint(tmp_path):
    output = tmp_path / "weights.pth"
    with pytest.raises(ValueError, match="no samples"):
        train.train_model(output, epochs=1, train_loader=[])
    assert not output.exists()
