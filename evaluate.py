# evaluate.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import load_model
from predict import _validate_logits

def validate_evaluation_batch(images, labels):
    if not torch.is_tensor(images) or images.dtype != torch.float32 or images.ndim not in (3, 4) or tuple(images.shape[1:]) not in ((28, 28), (1, 28, 28)) or not torch.isfinite(images).all():
        raise ValueError("Evaluation images must be finite float32 digit batches shaped N x 28 x 28 or N x 1 x 28 x 28")
    if not torch.is_tensor(labels) or labels.ndim != 1 or labels.dtype not in (torch.int32, torch.int64) or len(labels) != len(images) or not len(labels) or ((labels < 0) | (labels > 9)).any():
        raise ValueError("Evaluation labels must align with images and contain digit indexes 0 through 9")
    return labels.to(torch.int64)


def evaluate_model(model_path="model.pth", test_loader=None):
    model = load_model(model_path)
    if test_loader is None:
        transform = transforms.Compose([transforms.ToTensor()])
    
        test_data = datasets.MNIST(root=".", train=False, transform=transform, download=True)
        test_loader = DataLoader(test_data, batch_size=64)

    correct = 0
    total = 0
    confusion = torch.zeros((10, 10), dtype=torch.int64)

    with torch.no_grad():
        for images, labels in test_loader:
            labels = validate_evaluation_batch(images, labels)
            outputs = model(images)
            _validate_logits(outputs, (len(labels), 10))
            if labels.ndim != 1 or labels.dtype not in (torch.int32, torch.int64) or ((labels < 0) | (labels > 9)).any():
                raise ValueError("Evaluation labels must be digit indexes from 0 to 9")
            _, predicted = torch.max(outputs, 1)
            confusion += torch.bincount(labels.to(torch.int64) * 10 + predicted, minlength=100).reshape(10, 10)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        raise ValueError("Evaluation dataset contains no samples")

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return {"accuracy": accuracy, "samples": total, "correct": correct, "confusion_matrix": confusion.tolist()}

if __name__ == "__main__":
    evaluate_model()

