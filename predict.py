"""Run inference for handwritten digit images using a saved MNIST model."""

import torch
from itertools import islice
from PIL import Image, ImageOps
from torchvision import transforms

from model import load_model

NUM_CLASSES = 10
DEFAULT_BATCH_SIZE = 32
MAX_BATCH_SIZE = 1024


def preprocess_image(image_path, invert=False):
    """Convert an image to the 28x28 grayscale tensor expected by the model.

    Set invert=True for scans with dark ink on a light background when the
    trained model expects the usual MNIST light-digit-on-dark-background style.
    """
    with Image.open(image_path) as source_image:
        # Composite transparent pixels onto the model's dark background before
        # grayscale conversion. Otherwise hidden RGB values can become visible
        # and make transparent PNGs appear inverted or unexpectedly bright.
        rgba_image = ImageOps.exif_transpose(source_image).convert("RGBA")
        background_color = (255, 255, 255, 255) if invert else (0, 0, 0, 255)
        background = Image.new("RGBA", rgba_image.size, background_color)
        image = Image.alpha_composite(background, rgba_image).convert("L")
        if invert:
            image = ImageOps.invert(image)
        # Preserve the digit's aspect ratio and center it on the model canvas.
        # Stretching a wide or tall scan to 28x28 can distort the digit shape.
        image = ImageOps.contain(image, (28, 28), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (28, 28), color=0)
        offset = ((28 - image.width) // 2, (28 - image.height) // 2)
        canvas.paste(image, offset)
        image = canvas

    return transforms.ToTensor()(image).unsqueeze(0)


def _validate_top_k(top_k):
    """Require a candidate count that fits the ten MNIST output classes."""
    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or not 1 <= top_k <= NUM_CLASSES
    ):
        raise ValueError(f"top_k must be an integer between 1 and {NUM_CLASSES}.")


def _rank_predictions(probabilities, top_k):
    count = min(top_k, probabilities.numel())
    digits = torch.argsort(probabilities, descending=True, stable=True)[:count]
    scores = probabilities[digits]
    candidates = [
        {"digit": int(digit), "confidence": float(score)}
        for digit, score in zip(digits.tolist(), scores.tolist())
    ]
    return {
        "predicted_digit": candidates[0]["digit"],
        "confidence": candidates[0]["confidence"],
        "top_predictions": candidates,
    }


def _validate_logits(output, expected_shape):
    """Reject malformed or non-finite model output before applying softmax."""
    if output.ndim != 2 or tuple(output.shape) != expected_shape:
        raise ValueError(f"Model must return logits with shape {expected_shape}.")
    if not torch.isfinite(output).all():
        raise ValueError("Model returned non-finite logits.")


def predict(image_path, model_path="model.pth", top_k=3, invert=False, print_report=True):
    """Return the predicted digit, confidence, and top-k candidate probabilities."""
    _validate_top_k(top_k)

    model = load_model(model_path)
    tensor_image = preprocess_image(image_path, invert=invert)
    with torch.no_grad():
        output = model(tensor_image)
        _validate_logits(output, (1, 10))
        result = _rank_predictions(torch.softmax(output, dim=1)[0], top_k)

    if print_report:
        print(f"Predicted Digit: {result['predicted_digit']}")
        print(f"Confidence: {result['confidence']:.3f}")
        if len(result["top_predictions"]) > 1:
            alternatives = ", ".join(
                f"{item['digit']} ({item['confidence']:.3f})"
                for item in result["top_predictions"]
            )
            print(f"Top candidates: {alternatives}")
    return result


def _validate_batch_size(batch_size):
    """Require a positive integer to keep batch inference memory bounded."""
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or not 1 <= batch_size <= MAX_BATCH_SIZE
    ):
        raise ValueError(f"batch_size must be a positive integer no greater than {MAX_BATCH_SIZE}.")


def predict_batch(
    image_paths,
    model_path="model.pth",
    top_k=3,
    invert=False,
    batch_size=DEFAULT_BATCH_SIZE,
):
    """Predict multiple images in bounded chunks while preserving input order."""
    if isinstance(image_paths, (str, bytes)):
        raise ValueError("image_paths must be an iterable of image paths, not a single string.")
    _validate_top_k(top_k)
    _validate_batch_size(batch_size)
    iterator = iter(image_paths)
    first_chunk = list(islice(iterator, batch_size))
    if not first_chunk:
        raise ValueError("image_paths must contain at least one image path.")

    model = load_model(model_path)
    results = []
    with torch.no_grad():
        chunk_paths = first_chunk
        while chunk_paths:
            batch = torch.cat(
                [preprocess_image(path, invert=invert) for path in chunk_paths],
                dim=0,
            )
            output = model(batch)
            expected_shape = (len(chunk_paths), NUM_CLASSES)
            _validate_logits(output, expected_shape)
            probabilities = torch.softmax(output, dim=1)
            results.extend(
                _rank_predictions(row, top_k)
                for row in probabilities
            )
            chunk_paths = list(islice(iterator, batch_size))
    return results

def main(argv=None):
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Predict the digit shown in an image.")
    parser.add_argument("image_path", help="Path to an image containing a handwritten digit")
    parser.add_argument("--model", default="model.pth", help="Path to the trained model weights")
    parser.add_argument("--top-k", type=int, default=3, help="Number of ranked candidates to return")
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert light backgrounds with dark ink to match MNIST image polarity",
    )
    parser.add_argument("--json", action="store_true", help="Print only the structured digit prediction as JSON")
    args = parser.parse_args(argv)
    result = predict(args.image_path, model_path=args.model, top_k=args.top_k, invert=args.invert, **({"print_report": False} if args.json else {}))
    if args.json:
        print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

