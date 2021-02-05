import torch
import torch.nn as nn
import torch.nn.functional as F


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
        n, b = obs.shape[0:2]
        obs = torch.reshape(obs, (-1,) + obs.shape[2:])

        embed = self._encoder(obs)
        pred_obs = self._decoder(embed)

        pred_obs = torch.reshape(pred_obs, (n, b) + pred_obs.shape[1:])
        return pred_obs

    def loss(self, output, target):
        return self._decoder.loss(output, target)
