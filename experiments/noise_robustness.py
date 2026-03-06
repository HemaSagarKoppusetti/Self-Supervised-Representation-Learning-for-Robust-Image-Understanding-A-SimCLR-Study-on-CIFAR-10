import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class AddGaussianNoise(object):

    def __init__(self, std=0.2):
        self.std = std

    def __call__(self, tensor):

        noise = torch.randn_like(tensor) * self.std

        return tensor + noise


transform = transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise(std=0.3)
])


dataset = CIFAR10("./data", train=False, download=True, transform=transform)

loader = DataLoader(dataset, batch_size=16)

images, _ = next(iter(loader))

grid = images[:16]

plt.figure(figsize=(6,6))

for i in range(16):

    plt.subplot(4,4,i+1)

    img = grid[i].permute(1,2,0).numpy()

    plt.imshow(img)

    plt.axis("off")

plt.show()