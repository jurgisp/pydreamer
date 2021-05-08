import numpy as np
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


def cat(x1, x2):
    # (..., A), (..., B) => (..., A+B)
    return torch.cat((x1, x2), dim=-1)


def cat3(x1, x2, x3):
    return torch.cat((x1, x2, x3), dim=-1)


def split(mean_std, sizes=None):
    # (..., S+S) => (..., S), (..., S)
    if sizes == None:
        sizes = mean_std.size(-1) // 2
    return mean_std.split(sizes, dim=-1)


def diag_normal(mean_std):
    mean, std = split(mean_std)
    return D.independent.Independent(D.normal.Normal(mean, std), 1)


def to_mean_std(x, min_std):
    # DreamerV2:
    # std = {
    #     'softplus': lambda: tf.nn.softplus(std),
    #     'abs': lambda: tf.math.abs(std + 1),
    #     'sigmoid': lambda: tf.nn.sigmoid(std),
    #     'sigmoid2': lambda: 2 * tf.nn.sigmoid(std / 2),
    # }[self._std_act]()
    # std = std + self._min_std

    mean, std = split(x)
    std = F.softplus(std) + min_std
    # std = 2 * torch.sigmoid(std / 2) + min_std
    return cat(mean, std)


def zero_prior_like(mean_std):
    # Returns prior with 0 mean and unit variance
    mean, std = split(mean_std)
    prior = cat(torch.zeros_like(mean), torch.ones_like(std))
    return prior


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
