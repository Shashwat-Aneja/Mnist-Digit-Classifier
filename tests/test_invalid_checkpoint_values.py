import torch,pytest
from model import DigitClassifier,load_model
@pytest.mark.parametrize('value',[float('nan'),float('inf')])
def test_nonfinite_checkpoint_is_rejected_before_loading(tmp_path,value):
    state=DigitClassifier().state_dict();next(iter(state.values())).view(-1)[0]=value
    p=tmp_path/'bad.pth';torch.save(state,p)
    with pytest.raises(ValueError,match='finite real'):load_model(p)
def test_complex_checkpoint_does_not_silently_drop_imaginary_weights(tmp_path):
    state={k:v.to(torch.complex64)+1j for k,v in DigitClassifier().state_dict().items()}
    p=tmp_path/'bad.pth';torch.save(state,p)
    with pytest.raises(ValueError,match='finite real'):load_model(p)
