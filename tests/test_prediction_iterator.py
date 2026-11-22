import torch
import predict as app
def test_prediction_iterator_is_lazy_and_does_not_leak_no_grad(monkeypatch):
    consumed=[]
    def paths():
        for i in range(10):consumed.append(i);yield str(i)
    monkeypatch.setattr(app,'load_model',lambda _:lambda x:torch.zeros((len(x),10)))
    monkeypatch.setattr(app,'preprocess_image',lambda *a,**k:torch.zeros((1,1,28,28)))
    iterator=app.iter_predictions(paths(),batch_size=2)
    assert consumed==[]
    assert next(iterator)['predicted_digit']==0 and consumed==[0,1]
    assert torch.is_grad_enabled()
    next(iterator);assert consumed==[0,1]
    next(iterator);assert consumed==[0,1,2,3]
    iterator.close();assert torch.is_grad_enabled()
