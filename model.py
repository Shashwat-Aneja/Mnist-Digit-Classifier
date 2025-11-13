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
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
