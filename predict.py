# predict.py
import torch
from PIL import Image
from torchvision import transforms
from model import load_model

def predict(image_path, model_path="model.pth"):
    model = load_model(model_path)

    img = Image.open(image_path).convert("L")  # convert to grayscale
    img = img.resize((28, 28))                 # MNIST size

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    tensor_img = transform(img).unsqueeze(0)   # shape: (1, 1, 28, 28)

    with torch.no_grad():
        output = model(tensor_img)
        probabilities = torch.softmax(output, dim=1)
        predicted_digit = torch.argmax(probabilities).item()
        confidence = probabilities[0][predicted_digit].item()

    print(f"Predicted Digit: {predicted_digit}")
    print(f"Confidence: {confidence:.3f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
    else:
        predict(sys.argv[1])
