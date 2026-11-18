"""Train weights compatible with the shared digit-classifier loader."""
import argparse
from pathlib import Path
import os
import tempfile
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import DigitClassifier


def save_checkpoint_atomic(state_dict, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=destination.parent, suffix=".pth", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(state_dict, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def train_model(output_path="model.pth", epochs=5, train_loader=None):
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if train_loader is None:
        data = datasets.MNIST(root=".", train=True, transform=transforms.ToTensor(), download=True)
        train_loader = DataLoader(data, batch_size=64, shuffle=True)
    model = DigitClassifier()
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        samples = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            samples += len(labels)
        if samples == 0:
            raise ValueError("Training dataset contains no samples")
        print(f"Epoch {epoch + 1} completed.")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint_atomic(model.state_dict(), destination)
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the shared MNIST digit classifier")
    parser.add_argument("--output", default="model.pth")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args(argv)
    train_model(args.output, args.epochs)


if __name__ == "__main__":
    main()
