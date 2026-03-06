import torch
from models.simclr_model import SimCLR

model = SimCLR()

x = torch.randn(4, 3, 32, 32)

h, z = model(x)

print("Representation:", h.shape)
print("Projection:", z.shape)