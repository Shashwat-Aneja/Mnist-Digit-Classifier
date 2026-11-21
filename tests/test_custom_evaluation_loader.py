import torch
import evaluate
def test_supplied_digit_loader_is_evaluated_without_dataset_request(monkeypatch):
    monkeypatch.setattr(evaluate,'load_model',lambda _:lambda x:torch.nn.functional.one_hot(torch.tensor([1,2]),10).float())
    monkeypatch.setattr(evaluate.datasets,'MNIST',lambda **kwargs:(_ for _ in ()).throw(AssertionError('dataset download')))
    result=evaluate.evaluate_model('fixture',test_loader=[(torch.zeros((2,1,28,28)),torch.tensor([1,2]))])
    assert result['samples']==2 and result['correct']==2 and result['accuracy']==100
    assert result['confusion_matrix'][1][1]==1
