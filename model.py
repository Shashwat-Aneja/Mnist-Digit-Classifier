# model.py
import torch
from pathlib import Path

MAX_CHECKPOINT_BYTES = 16 * 1024 * 1024
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
    if isinstance(path, (str, Path)) and Path(path).is_file() and Path(path).stat().st_size > MAX_CHECKPOINT_BYTES:
        raise ValueError("Digit checkpoint exceeds the supported 16 MiB file budget")
    model = DigitClassifier()
    weights = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(weights, dict) or any(not torch.is_tensor(value) or value.is_complex() or not torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("Model weights must be finite real tensors")
    model.load_state_dict(weights)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Model weights exceed the destination tensor numeric range")
    model.eval()
    return model

