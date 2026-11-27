import torch
import pytest
import evaluate

def test_per_class_metrics_and_silent_report(monkeypatch,capsys):
    class Model:
        def __call__(self,images):
            return torch.nn.functional.one_hot(torch.tensor([0,0,1]),num_classes=10).float()
    monkeypatch.setattr(evaluate,"load_model",lambda p:Model())
    result=evaluate.evaluate_model(test_loader=[(torch.zeros(3,1,28,28),torch.tensor([0,1,1]))],print_report=False)
    assert result["per_class"][0]["precision"]==.5
    assert result["per_class"][1]["recall"]==.5
    assert result["per_class"][2]["recall"] is None
    assert result["balanced_accuracy_percent"]==pytest.approx(75)
    assert capsys.readouterr().out==""
