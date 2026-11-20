# model.py
import torch
from torch import nn

class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)


def load_model(path="model.pth"):
    model = DigitClassifier()
    weights = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(weights, dict) or any(not torch.is_tensor(value) or value.is_complex() or not torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("Model weights must be finite real tensors")
    model.load_state_dict(weights)
    model.eval()
    return model

