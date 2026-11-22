import torch,pytest
from train import train_model
@pytest.mark.parametrize('images,labels',[(torch.zeros((2,1,20,20)),torch.tensor([0,1])),(torch.zeros((2,1,28,28)),torch.tensor([0])),(torch.zeros((2,1,28,28)),torch.tensor([0,10])),(torch.zeros((2,1,28,28)),torch.tensor([False,True]))])
def test_invalid_training_batch_preserves_previous_checkpoint(tmp_path,images,labels):
    p=tmp_path/'weights.pth';p.write_bytes(b'keep')
    with pytest.raises(ValueError):train_model(p,epochs=1,train_loader=[(images,labels)])
    assert p.read_bytes()==b'keep'
def test_int32_training_labels_are_normalized(tmp_path):
    p=tmp_path/'weights.pth';train_model(p,epochs=1,train_loader=[(torch.zeros((2,1,28,28)),torch.tensor([0,1],dtype=torch.int32))])
    assert p.exists()
