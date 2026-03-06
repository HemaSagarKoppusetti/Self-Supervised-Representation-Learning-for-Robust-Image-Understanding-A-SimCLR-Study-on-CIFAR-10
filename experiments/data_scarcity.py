import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from models.simclr_model import SimCLR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_subset(dataset, fraction):

    size = int(len(dataset) * fraction)

    indices = torch.randperm(len(dataset))[:size]

    return Subset(dataset, indices)


def load_encoder():

    model = SimCLR()

    model.load_state_dict(torch.load("results/simclr_epoch_50.pth"))

    encoder = model.encoder

    for p in encoder.parameters():
        p.requires_grad = False

    encoder.to(DEVICE)

    return encoder


class LinearClassifier(nn.Module):

    def __init__(self, encoder):
        super().__init__()

        self.encoder = encoder
        self.fc = nn.Linear(512, 10)

    def forward(self, x):

        with torch.no_grad():
            h = self.encoder(x)
            h = torch.flatten(h, 1)

        return self.fc(h)


def run_experiment(data_fraction):

    transform = transforms.ToTensor()

    train_dataset = CIFAR10("./data", train=True, download=True, transform=transform)

    train_subset = get_subset(train_dataset, data_fraction)

    loader = DataLoader(train_subset, batch_size=128, shuffle=True)

    encoder = load_encoder()

    model = LinearClassifier(encoder).to(DEVICE)

    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):

        for x, y in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)

            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(f"Training finished with {data_fraction*100}% labels")


if __name__ == "__main__":

    for fraction in [0.1, 0.5, 1.0]:

        run_experiment(fraction)