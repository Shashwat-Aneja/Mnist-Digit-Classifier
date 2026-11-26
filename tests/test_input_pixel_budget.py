from PIL import Image
import pytest
import predict

def test_pixel_budget_precedes_exif_and_rgba_allocations(tmp_path,monkeypatch):
    p=tmp_path/'in.png';Image.new('RGB',(3,3)).save(p)
    monkeypatch.setattr(predict,'MAX_INPUT_IMAGE_PIXELS',4)
    def unexpected(*args):raise AssertionError('large image was expanded')
    monkeypatch.setattr(predict.ImageOps,'exif_transpose',unexpected)
    with pytest.raises(ValueError,match='pixel budget'):predict.preprocess_image(p)

def test_exact_pixel_budget_is_accepted(tmp_path,monkeypatch):
    p=tmp_path/'in.png';Image.new('RGB',(2,2)).save(p)
    monkeypatch.setattr(predict,'MAX_INPUT_IMAGE_PIXELS',4)
    assert tuple(predict.preprocess_image(p).shape)==(1,1,28,28)
