import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset


class SimCLRTransform:
    """
    Creates two augmented views of the same image.
    """

    def __init__(self, size=32):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class CIFAR10Pair(Dataset):
    """
    Returns two augmented views of the same CIFAR image.
    """

    def __init__(self, root="./data", train=True):
        self.dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=True
        )
        self.transform = SimCLRTransform()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]

        x_i, x_j = self.transform(img)

        return x_i, x_j