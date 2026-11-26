import pytest,torch
from predict import _validate_logits

@pytest.mark.parametrize('output',[None,[[0]*10],torch.zeros(1,10,dtype=torch.int64),torch.zeros(1,10,dtype=torch.complex64)])
def test_invalid_model_output_types_are_rejected_before_softmax(output):
    with pytest.raises(ValueError,match='floating-point tensor'):_validate_logits(output,(1,10))
