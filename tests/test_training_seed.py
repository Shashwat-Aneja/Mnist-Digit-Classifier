import torch,pytest
from train import train_model
def test_same_seed_repeats_synthetic_cpu_training_weights(tmp_path):
    loader=[(torch.zeros((2,1,28,28)),torch.tensor([0,1]))]
    with torch.random.fork_rng():
        first=train_model(tmp_path/'one.pth',epochs=1,train_loader=loader,seed=7)
        second=train_model(tmp_path/'two.pth',epochs=1,train_loader=loader,seed=7)
    assert all(torch.equal(value,second.state_dict()[key]) for key,value in first.state_dict().items())
def test_invalid_seed_fails_before_dataset_creation(tmp_path):
    with pytest.raises(ValueError,match='seed'):train_model(tmp_path/'bad.pth',seed=True)
    assert not (tmp_path/'bad.pth').exists()
