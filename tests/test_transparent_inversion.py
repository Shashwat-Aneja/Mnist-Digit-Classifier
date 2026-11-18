from PIL import Image
from predict import preprocess_image


def test_inverting_transparent_dark_ink_keeps_background_dark(tmp_path):
    image = Image.new("RGBA", (28, 28), (50, 100, 200, 0))
    image.putpixel((14, 14), (0, 0, 0, 255))
    source = tmp_path / "digit.png"
    image.save(source)
    result = preprocess_image(source, invert=True)
    assert result[0, 0, 0, 0].item() == 0
    assert result[0, 0, 14, 14].item() == 1
