from PIL import Image
from predict import preprocess_image
def test_small_square_scan_is_scaled_to_model_canvas(tmp_path):
    p=tmp_path/'small.png';Image.new('L',(4,4),255).save(p)
    tensor=preprocess_image(p)
    assert tuple(tensor.shape)==(1,1,28,28) and tensor.min().item()>.99
def test_small_wide_scan_preserves_aspect_ratio(tmp_path):
    p=tmp_path/'wide.png';Image.new('L',(8,4),255).save(p)
    tensor=preprocess_image(p)[0,0]
    assert tensor[0].max().item()==0 and tensor[14].min().item()>.99
