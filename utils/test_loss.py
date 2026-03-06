import torch
from utils.contrastive_loss import NTXentLoss

batch_size = 4

z_i = torch.randn(batch_size, 128)
z_j = torch.randn(batch_size, 128)

criterion = NTXentLoss(batch_size)

loss = criterion(z_i, z_j)

print(loss)