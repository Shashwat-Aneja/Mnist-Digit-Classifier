import torch
from train import train_model

def test_seeded_training_restores_caller_cpu_random_state(tmp_path):
    images=torch.zeros(2,1,28,28);labels=torch.tensor([1,2]);torch.manual_seed(123)
    before=torch.random.get_rng_state().clone()
    train_model(tmp_path/'model.pth',epochs=1,train_loader=[(images,labels)],seed=7)
    assert torch.equal(before,torch.random.get_rng_state())
