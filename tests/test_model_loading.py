"""Regression tests for safe MNIST checkpoint loading."""

import torch

from model import DigitClassifier, load_model


def test_load_model_restores_state_and_enters_eval_mode(tmp_path):
    checkpoint = tmp_path / "weights.pth"
    original = DigitClassifier()
    torch.save(original.state_dict(), checkpoint)

    restored = load_model(checkpoint)

    assert not restored.training
    for name, expected in original.state_dict().items():
        assert torch.equal(restored.state_dict()[name], expected)


def test_load_model_requests_weights_only_deserialization(monkeypatch):
    observed = {}
    original_load = torch.load

    def recording_load(*args, **kwargs):
        observed.update(kwargs)
        return DigitClassifier().state_dict()

    monkeypatch.setattr(torch, "load", recording_load)
    load_model("unused.pth")

    assert observed["weights_only"] is True
    assert observed["map_location"] == "cpu"
