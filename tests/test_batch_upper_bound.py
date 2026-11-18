import pytest
from predict import predict_batch, MAX_BATCH_SIZE


@pytest.mark.parametrize("batch_size", [MAX_BATCH_SIZE + 1, 2**64])
def test_oversized_batch_is_rejected_before_consuming_paths(batch_size):
    def paths():
        raise AssertionError("invalid batch must not consume input paths")
        yield "unused.png"
    with pytest.raises(ValueError, match="no greater"):
        predict_batch(paths(), batch_size=batch_size)
