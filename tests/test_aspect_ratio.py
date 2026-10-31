from PIL import Image, ImageDraw

import predict as predictor


def test_preprocess_preserves_landscape_aspect_ratio_and_centers_digit(tmp_path):
    image_path = tmp_path / "landscape.png"
    image = Image.new("L", (56, 28), color=0)
    ImageDraw.Draw(image).rectangle((20, 4, 35, 23), fill=255)
    image.save(image_path)

    tensor = predictor.preprocess_image(str(image_path))

    assert tuple(tensor.shape) == (1, 1, 28, 28)
    assert tensor[0, 0, 0, :].max().item() == 0.0
    assert tensor[0, 0, 27, :].max().item() == 0.0
    assert tensor[0, 0, 14, 14].item() > 0.9


def test_preprocess_preserves_portrait_aspect_ratio_and_centers_digit(tmp_path):
    image_path = tmp_path / "portrait.png"
    image = Image.new("L", (28, 56), color=0)
    ImageDraw.Draw(image).rectangle((4, 20, 23, 35), fill=255)
    image.save(image_path)

    tensor = predictor.preprocess_image(str(image_path))

    assert tuple(tensor.shape) == (1, 1, 28, 28)
    assert tensor[0, 0, :, 0].max().item() == 0.0
    assert tensor[0, 0, :, 27].max().item() == 0.0
    assert tensor[0, 0, 14, 14].item() > 0.9
