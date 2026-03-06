import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    NT-Xent loss used in SimCLR
    """

    def __init__(self, batch_size, temperature=0.5):
        super().__init__()

        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self._create_mask(batch_size)

    def _create_mask(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)

        mask.fill_diagonal_(0)

        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0

        return mask

    def forward(self, z_i, z_j):

        batch_size = z_i.shape[0]

        z = torch.cat([z_i, z_j], dim=0)

        z = F.normalize(z, dim=1)

        similarity_matrix = torch.matmul(z, z.T)

        N = 2 * batch_size

        positives = torch.cat([
            torch.diag(similarity_matrix, batch_size),
            torch.diag(similarity_matrix, -batch_size)
        ])

        negatives = similarity_matrix[self.mask].view(N, -1)

        logits = torch.cat((positives.unsqueeze(1), negatives), dim=1)

        logits = logits / self.temperature

        labels = torch.zeros(N).long().to(z.device)

        loss = F.cross_entropy(logits, labels)

        return loss