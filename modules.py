import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


def flatten(x):
    return torch.reshape(x, (-1,) + x.shape[2:]), x.size(0)


def unflatten(x, n):
    return torch.reshape(x, (n, -1) + x.shape[1:])


def diag_normal(mean, std):
    return D.independent.Independent(D.normal.Normal(mean, std), 1)


class Posterior(nn.Module):

    def __init__(self, in1_dim=256, in2_dim=256, hidden_dim=256, out_dim=30):
        super().__init__()
        self._model = nn.Sequential(
            nn.Linear(in1_dim + in2_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * out_dim))

    def forward(self, deter, embed):
        mean_std = self._model(torch.cat((deter, embed), dim=-1))
        mean, std = mean_std.chunk(chunks=2, dim=-1)
        std = F.softplus(std) + 0.1
        return mean, std


class MinigridEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self._model = nn.Sequential(
            # nn.Conv2d(20, 4, kernel_size=1),  # embedding
            nn.Flatten(-3, -1),
            nn.Linear(20 * 7 * 7, 256),
            nn.ReLU()
        )
        self.out_dim = 256

    def forward(self, x):
        return self._model(x)


class MinigridDecoder(nn.Module):

    def __init__(self):
        super().__init__()
        self._model = nn.Sequential(
            # nn.Linear(256, 256),
            # nn.ReLU(),
            nn.Linear(256, 20 * 7 * 7),
            nn.Unflatten(-1, (20, 7, 7)),
            # nn.Conv2d(4, 20, kernel_size=1)
        )
        self.in_dim = 256

    def forward(self, x):
        return self._model(x)

    def loss(self, output, target):
        loss = F.binary_cross_entropy_with_logits(output, target, reduction='none')
        return loss.sum(dim=[-1, -2, -3])
