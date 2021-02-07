import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


def flatten(x):
    # (N, B, ...) => (N*B, ...)
    return torch.reshape(x, (-1,) + x.shape[2:]), x.size(0)


def unflatten(x, n):
    # (N*B, ...) => (N, B, ...)
    return torch.reshape(x, (n, -1) + x.shape[1:])


def diag_normal(mean, std):
    return D.independent.Independent(D.normal.Normal(mean, std), 1)


class Posterior(nn.Module):

    def __init__(self, in1_dim=256, in2_dim=256, hidden_dim=256, out_dim=30, min_std=0.1):
        super().__init__()
        self._model = nn.Sequential(
            nn.Linear(in1_dim + in2_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * out_dim))
        self._min_std = min_std

    def forward(self, deter, embed):
        mean_std = self._model(torch.cat((deter, embed), dim=-1))
        mean, std = mean_std.chunk(chunks=2, dim=-1)
        std = F.softplus(std) + self._min_std
        return mean, std


class MinigridEncoder(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self._model = nn.Sequential(
            nn.Flatten(-3, -1),
            nn.Linear(in_channels * 7 * 7, 256),
            nn.ELU()
        )
        self.out_dim = 256

    def forward(self, x):
        return self._model(x)


class MinigridDecoderBCE(nn.Module):

    def __init__(self, in_dim=256):
        super().__init__()
        self._model = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 20 * 7 * 7),
            nn.Unflatten(-1, (20, 7, 7)),
        )
        self.in_dim = in_dim

    def forward(self, x):
        return self._model(x)

    def loss(self, output, target):
        loss = F.binary_cross_entropy_with_logits(output, target, reduction='none')
        return loss.sum(dim=[-1, -2, -3])


class MinigridDecoderCE(nn.Module):

    def __init__(self, in_dim=256):
        super().__init__()
        self._model = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 33 * 7 * 7),
            nn.Unflatten(-1, (33, 7, 7)),
        )
        self.in_dim = in_dim

    def forward(self, x):
        return self._model(x)

    def loss(self, output, target):
        output, n = flatten(output)
        target, _ = flatten(target)
        loss = F.cross_entropy(output, target.argmax(dim=-3), reduction='none')
        loss = unflatten(loss, n)
        return loss.sum(dim=[-1, -2])