import torch
import torch.nn as nn
import torch.nn.functional as F


def flatten(x):
    return torch.reshape(x, (-1,) + x.shape[2:]), x.size(0)

def unflatten(x, n):
    return torch.reshape(x, (n, -1) + x.shape[1:])

class Autoencoder(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder

    def forward(self, obs, action):
        # Input:
        #   obs: tensor(N, B, C, H, W)
        # Output:
        # 
        obs, n = flatten(obs)

        embed = self._encoder(obs)
        pred_obs = self._decoder(embed)

        pred_obs = unflatten(pred_obs, n)
        return pred_obs

    def loss(self, output, target):
        return self._decoder.loss(output, target)
