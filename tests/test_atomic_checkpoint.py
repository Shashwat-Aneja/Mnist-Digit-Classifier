import pytest
import torch
import train


def test_failed_checkpoint_save_preserves_previous_weights(tmp_path, monkeypatch):
    destination = tmp_path / "model.pth"
    destination.write_bytes(b"previous checkpoint")
    def fail(state, handle):
        handle.write(b"partial")
        raise OSError("disk full")
    monkeypatch.setattr(train.torch, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        train.save_checkpoint_atomic({}, destination)
    assert destination.read_bytes() == b"previous checkpoint"
    assert list(tmp_path.iterdir()) == [destination]


def test_successful_atomic_checkpoint_is_loadable(tmp_path):
    destination = tmp_path / "model.pth"
    train.save_checkpoint_atomic({"weights": torch.tensor([1, 2])}, destination)
    assert torch.equal(torch.load(destination, weights_only=True)["weights"], torch.tensor([1, 2]))
