import torch
from predict import _rank_predictions
def test_equal_digit_probabilities_are_ranked_by_digit_index():
    result=_rank_predictions(torch.full((10,),.1),10)
    assert result['predicted_digit']==0
    assert [r['digit'] for r in result['top_predictions']]==list(range(10))
