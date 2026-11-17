import torch
from PIL import Image, ImageOps
from predict import preprocess_image


def test_exif_orientation_is_applied_before_resizing(tmp_path):
    image = Image.new("L", (14, 28), 0)
    for x in range(7):
        for y in range(28):
            image.putpixel((x, y), 255)
    exif = image.getexif()
    exif[274] = 6
    tagged = tmp_path / "tagged.png"
    image.save(tagged, exif=exif)
    with Image.open(tagged) as source:
        oriented = ImageOps.exif_transpose(source)
        reference = tmp_path / "oriented.png"
        oriented.save(reference)
    assert torch.equal(preprocess_image(tagged), preprocess_image(reference))
