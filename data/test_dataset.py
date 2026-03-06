from dataset import CIFAR10Pair
from torch.utils.data import DataLoader

dataset = CIFAR10Pair()
loader = DataLoader(dataset, batch_size=4, shuffle=True)

for x_i, x_j in loader:
    print(x_i.shape, x_j.shape)
    break