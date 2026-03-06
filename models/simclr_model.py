import torch
import torch.nn as nn
import torchvision.models as models


class ProjectionHead(nn.Module):
    """
    MLP projection head used in SimCLR
    """

    def __init__(self, input_dim=512, hidden_dim=512, output_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class SimCLR(nn.Module):
    """
    SimCLR Model
    """

    def __init__(self):
        super().__init__()

        # Encoder: ResNet18
        resnet = models.resnet18(pretrained=False)

        # Remove classification head
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

        feature_dim = resnet.fc.in_features

        # Projection head
        self.projector = ProjectionHead(feature_dim)

    def forward(self, x):

        h = self.encoder(x)

        h = torch.flatten(h, start_dim=1)

        z = self.projector(h)

        return h, z