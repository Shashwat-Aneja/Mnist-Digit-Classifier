import pytest,torch
import evaluate

@pytest.mark.parametrize('images,labels',[(torch.zeros(2,1,28,28),torch.tensor([1])),(torch.zeros(2,3,28,28),torch.tensor([1,2])),(torch.full((2,1,28,28),float('nan')),torch.tensor([1,2])),(torch.zeros(2,1,28,28,dtype=torch.float64),torch.tensor([1,2])),(torch.zeros(2,1,28,28),[1,2])])
def test_bad_evaluation_batches_fail_before_forward(monkeypatch,images,labels):
    class Model:
        def __call__(self,*args):raise AssertionError('invalid batch reached model')
    monkeypatch.setattr(evaluate,'load_model',lambda path:Model())
    with pytest.raises(ValueError,match='Evaluation'):evaluate.evaluate_model(test_loader=[(images,labels)])
