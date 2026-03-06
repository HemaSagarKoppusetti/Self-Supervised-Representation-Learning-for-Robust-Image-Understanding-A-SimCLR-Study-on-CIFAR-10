import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import CIFAR10Pair
from models.simclr_model import SimCLR
from utils.contrastive_loss import NTXentLoss


# Training parameters
BATCH_SIZE = 128
EPOCHS = 50
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train():

    # Dataset
    dataset = CIFAR10Pair()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Model
    model = SimCLR().to(DEVICE)

    # Loss
    criterion = NTXentLoss(BATCH_SIZE).to(DEVICE)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        loop = tqdm(loader)

        for x_i, x_j in loop:

            x_i = x_i.to(DEVICE)
            x_j = x_j.to(DEVICE)

            _, z_i = model(x_i)
            _, z_j = model(x_j)

            loss = criterion(z_i, z_j)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)

        print(f"Epoch {epoch+1} Loss: {avg_loss}")

        # Save checkpoint
        torch.save(model.state_dict(), f"results/simclr_epoch_{epoch+1}.pth")


if __name__ == "__main__":
    train()