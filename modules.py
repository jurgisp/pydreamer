import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


def flatten(x):
    # (N, B, ...) => (N*B, ...)
    return torch.reshape(x, (-1,) + x.shape[2:])


def unflatten(x, n):
    # (N*B, ...) => (N, B, ...)
    return torch.reshape(x, (n, -1) + x.shape[1:])


def diag_normal(mean_std):
    mean, std = split(mean_std)
    return D.independent.Independent(D.normal.Normal(mean, std), 1)


def zero_prior_like(mean_std):
    # Returns prior with 0 mean and unit variance
    mean, std = split(mean_std)
    prior = join(torch.zeros_like(mean), torch.ones_like(std))
    return prior


def join(mean, std):
    mean_std = torch.cat((mean, std), dim=-1)
    return mean_std


def split(mean_std):
    mean, std = mean_std.chunk(chunks=2, dim=-1)
    return mean, std


class Posterior(nn.Module):

    def __init__(self, in1_dim=256, in2_dim=256, hidden_dim=256, out_dim=30, min_std=0.1):
        super().__init__()
        self._model = nn.Sequential(
            nn.Linear(in1_dim + in2_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * out_dim))
        self._min_std = min_std

    def forward(self,
                deter,  # tensor(..., D)
                embed,  # tensor(..., E)
                ):
        mean, std = split(self._model(join(deter, embed)))
        std = F.softplus(std) + self._min_std
        return (
            join(mean, std)  # tensor(..., 2*S)
        )


class MinigridEncoder(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self._model = nn.Sequential(
            nn.Flatten(-3, -1),
            nn.Linear(in_channels * 7 * 7, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
        )
        self.out_dim = 256

    def forward(self, x):
        return self._model(x)


class MinigridDecoderCE(nn.Module):

    def __init__(self, in_dim=30):
        super().__init__()
        self._model = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 33 * 7 * 7),
            nn.Unflatten(-1, (33, 7, 7)),
        )
        self.in_dim = in_dim

    def forward(self, x):
        return self._model(x)

    def loss(self, output, target):
        n = output.size(0)
        output = flatten(output)
        target = flatten(target).argmax(dim=-3)
        loss = F.cross_entropy(output, target, reduction='none')
        loss = unflatten(loss, n)
        return loss.sum(dim=[-1, -2])
