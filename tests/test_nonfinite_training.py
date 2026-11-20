import torch,pytest
import train
def test_nonfinite_images_do_not_replace_saved_weights(tmp_path):
    p=tmp_path/'weights.pth';p.write_bytes(b'keep')
    data=[(torch.full((2,1,28,28),float('nan')),torch.tensor([0,1]))]
    with pytest.raises(ValueError,match='finite'):train.train_model(p,epochs=1,train_loader=data)
    assert p.read_bytes()==b'keep'
def test_nonfinite_logits_do_not_save_checkpoint(tmp_path,monkeypatch):
    monkeypatch.setattr(train.DigitClassifier,'forward',lambda self,x:torch.full((len(x),10),float('inf')))
    p=tmp_path/'weights.pth'
    with pytest.raises(ValueError,match='logits'):train.train_model(p,epochs=1,train_loader=[(torch.zeros((2,1,28,28)),torch.tensor([0,1]))])
    assert not p.exists()

def test_optimizer_corruption_does_not_replace_checkpoint(tmp_path,monkeypatch):
    def corrupt(self,*args,**kwargs):
        with torch.no_grad():self.param_groups[0]['params'][0].fill_(float('inf'))
    monkeypatch.setattr(torch.optim.Adam,'step',corrupt)
    p=tmp_path/'weights.pth';p.write_bytes(b'keep')
    with pytest.raises(ValueError,match='weights'):train.train_model(p,epochs=1,train_loader=[(torch.zeros((2,1,28,28)),torch.tensor([0,1]))])
    assert p.read_bytes()==b'keep'
