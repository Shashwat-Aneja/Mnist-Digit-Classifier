"""Run inference for a handwritten digit image using a saved MNIST model."""

import torch
from PIL import Image, ImageOps
from torchvision import transforms

from model import load_model


def preprocess_image(image_path, invert=False):
    """Convert an image to the 28x28 grayscale tensor expected by the model.

    Set invert=True for scans with dark ink on a light background when the
    trained model expects the usual MNIST light-digit-on-dark-background style.
    """
    with Image.open(image_path) as source_image:
        image = source_image.convert("L")
        if invert:
            image = ImageOps.invert(image)
        image = image.resize((28, 28))

    return transforms.ToTensor()(image).unsqueeze(0)


def predict(image_path, model_path="model.pth", top_k=3, invert=False):
    """Return the predicted digit, confidence, and top-k candidate probabilities."""
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    model = load_model(model_path)
    tensor_image = preprocess_image(image_path, invert=invert)
    with torch.no_grad():
        output = model(tensor_image)
        if output.ndim != 2 or output.shape[0] != 1 or output.shape[1] != 10:
            raise ValueError("Model must return logits with shape (1, 10).")
        probabilities = torch.softmax(output, dim=1)[0]
        count = min(top_k, probabilities.numel())
        scores, digits = torch.topk(probabilities, k=count)

    candidates = [
        {"digit": int(digit), "confidence": float(score)}
        for digit, score in zip(digits.tolist(), scores.tolist())
    ]
    result = {
        "predicted_digit": candidates[0]["digit"],
        "confidence": candidates[0]["confidence"],
        "top_predictions": candidates,
    }

    print(f"Predicted Digit: {result['predicted_digit']}")
    print(f"Confidence: {result['confidence']:.3f}")
    if len(candidates) > 1:
        alternatives = ", ".join(
            f"{item['digit']} ({item['confidence']:.3f})" for item in candidates
        )
        print(f"Top candidates: {alternatives}")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict the digit shown in an image.")
    parser.add_argument("image_path", help="Path to an image containing a handwritten digit")
    parser.add_argument("--model", default="model.pth", help="Path to the trained model weights")
    parser.add_argument("--top-k", type=int, default=3, help="Number of ranked candidates to return")
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert light backgrounds with dark ink to match MNIST image polarity",
    )
    args = parser.parse_args()
    predict(args.image_path, model_path=args.model, top_k=args.top_k, invert=args.invert)
