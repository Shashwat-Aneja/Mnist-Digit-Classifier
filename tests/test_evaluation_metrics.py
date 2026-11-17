import torch
import evaluate


def test_evaluation_returns_confusion_matrix_and_accuracy(monkeypatch):
    images = torch.zeros((3, 1, 28, 28))
    labels = torch.tensor([0, 1, 2])
    monkeypatch.setattr(evaluate.datasets, "MNIST", lambda **kwargs: [])
    monkeypatch.setattr(evaluate, "DataLoader", lambda *args, **kwargs: [(images, labels)])
    class Model:
        def __call__(self, batch):
            logits = torch.zeros((3, 10))
            logits[0, 0] = 1
            logits[1, 1] = 1
            logits[2, 1] = 1
            return logits
    monkeypatch.setattr(evaluate, "load_model", lambda path: Model())
    result = evaluate.evaluate_model("unused")
    assert result["samples"] == 3
    assert result["correct"] == 2
    assert result["confusion_matrix"][2][1] == 1
    assert sum(map(sum, result["confusion_matrix"])) == 3
