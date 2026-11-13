"""Evaluation regression tests for empty datasets and checkpoint loading."""

import pytest

import evaluate


def test_evaluate_rejects_empty_test_dataset(monkeypatch):
    monkeypatch.setattr(evaluate.datasets, "MNIST", lambda **kwargs: [])
    monkeypatch.setattr(evaluate, "load_model", lambda path: object())

    with pytest.raises(ValueError, match="no samples"):
        evaluate.evaluate_model("unused.pth")


def test_evaluate_uses_shared_checkpoint_loader(monkeypatch):
    paths = []
    monkeypatch.setattr(evaluate.datasets, "MNIST", lambda **kwargs: [])

    def record_loader(path):
        paths.append(path)
        return object()

    monkeypatch.setattr(evaluate, "load_model", record_loader)
    with pytest.raises(ValueError, match="no samples"):
        evaluate.evaluate_model("safe-weights.pth")
    assert paths == ["safe-weights.pth"]
