from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch import Tensor


def flatten(x: Tensor) -> Tensor:
    # (N, B, ...) => (N*B, ...)
    return torch.reshape(x, (-1,) + x.shape[2:])


def flatten3(x: Tensor) -> Tensor:
    # (N, B, ...) => (N*B, ...)
    return torch.reshape(x, (-1,) + x.shape[3:])


def unflatten(x: Tensor, n: int) -> Tensor:
    # (N*B, ...) => (N, B, ...)
    return torch.reshape(x, (n, -1) + x.shape[1:])


def unflatten3(x: Tensor, nb: Tuple) -> Tensor:
    # (NBI, ...) => (N,B,I, ...)
    return torch.reshape(x, nb + (-1,) + x.shape[1:])


def cat(x1, x2):
    # (..., A), (..., B) => (..., A+B)
    return torch.cat((x1, x2), dim=-1)


def diag_normal(x: Tensor, min_std=0.1, max_std=2.0):
    # DreamerV2:
    # std = {
    #     'softplus': lambda: tf.nn.softplus(std),
    #     'sigmoid2': lambda: 2 * tf.nn.sigmoid(std / 2),
    # }[self._std_act]()
    # std = std + self._min_std

    mean, std = x.chunk(2, -1)
    # std = F.softplus(std) + min_std
    std = max_std * torch.sigmoid(std) + min_std
    return D.independent.Independent(D.normal.Normal(mean, std), 1)


def rsample(x: Tensor, noise: Tensor, min_std=0.1, max_std=2.0):
    mean, std = x.chunk(2, -1)
    std = max_std * torch.sigmoid(std) + min_std
    return mean + noise * std


def init_weights_tf2(m):
    # Match TF2 initializations
    if type(m) in {nn.Conv2d, nn.ConvTranspose2d, nn.Linear}:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    if type(m) == nn.GRUCell:
        nn.init.xavier_uniform_(m.weight_ih.data)
        nn.init.orthogonal_(m.weight_hh.data)
        nn.init.zeros_(m.bias_ih.data)
        nn.init.zeros_(m.bias_hh.data)


def imgrec_to_distr(x: Tensor) -> D.Categorical:  # (N,B,I,C,H,W) -> (N,B,H,W,C)
    assert len(x.shape) == 6
    logits = x.permute(2, 0, 1, 4, 5, 3)  # (N,B,I,C,H,W) => (I,N,B,H,W,C)
    # Normalize probability
    logits = logits - logits.logsumexp(dim=-1, keepdim=True)
    # Aggregate prob=avg(prob_i)
    logits_agg = torch.logsumexp(logits, dim=0)  # (I,N,B,H,W,C) => (N,B,H,W,C)
    return D.Categorical(logits=logits_agg)


def logavgexp(x: Tensor, dim: int) -> Tensor:
    return x.logsumexp(dim=dim) - np.log(x.size(dim))
