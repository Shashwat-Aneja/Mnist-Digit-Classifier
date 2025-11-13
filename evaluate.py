# evaluate.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import DigitClassifier

def evaluate_model(model_path="model.pth"):
    transform = transforms.Compose([transforms.ToTensor()])

    test_data = datasets.MNIST(root=".", train=False, transform=transform, download=True)
    test_loader = DataLoader(test_data, batch_size=64)

    model = DigitClassifier()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    evaluate_model()
