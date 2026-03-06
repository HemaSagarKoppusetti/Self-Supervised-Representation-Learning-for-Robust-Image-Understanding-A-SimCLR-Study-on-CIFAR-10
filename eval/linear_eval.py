import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader

from models.simclr_model import SimCLR


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
EPOCHS = 20


def load_encoder():

    model = SimCLR()

    model.load_state_dict(torch.load("results/simclr_epoch_50.pth"))

    encoder = model.encoder

    for param in encoder.parameters():
        param.requires_grad = False

    encoder.to(DEVICE)

    return encoder


def get_dataloaders():

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    return train_loader, test_loader


class LinearClassifier(nn.Module):

    def __init__(self, encoder):
        super().__init__()

        self.encoder = encoder

        self.fc = nn.Linear(512, 10)

    def forward(self, x):

        with torch.no_grad():
            h = self.encoder(x)
            h = torch.flatten(h, start_dim=1)

        out = self.fc(h)

        return out


def train():

    encoder = load_encoder()

    train_loader, test_loader = get_dataloaders()

    model = LinearClassifier(encoder).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):

        model.train()

        for x, y in train_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            preds = model(x)

            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = evaluate(model, test_loader)

        print(f"Epoch {epoch+1} Test Accuracy: {acc:.2f}%")


def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            preds = model(x)

            _, predicted = preds.max(1)

            total += y.size(0)
            correct += predicted.eq(y).sum().item()

    return 100 * correct / total


if __name__ == "__main__":
    train()