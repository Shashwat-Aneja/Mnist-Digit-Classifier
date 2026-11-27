import torch
import pytest
from model import DigitClassifier, load_model

def test_finite_float64_checkpoint_must_remain_finite_after_loading(tmp_path):
    path=tmp_path/"model.pth"
    weights={name: torch.full_like(value,dtype=torch.float64,fill_value=1e100) for name,value in DigitClassifier().state_dict().items()}
    assert all(torch.isfinite(value).all() for value in weights.values())
    torch.save(weights,path)
    with pytest.raises(ValueError,match="destination tensor"):
        load_model(path)
