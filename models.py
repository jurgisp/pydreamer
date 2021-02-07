import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import *


class VAE(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._posterior = Posterior(encoder.out_dim, encoder.out_dim, 256, decoder.in_dim)

    def forward(self,
                obs,  # tensor(N, B, C, H, W)
                action):

        obs, n = flatten(obs)
        embed = self._encoder(obs)

        deter = torch.zeros_like(embed)
        post_mean, post_std = self._posterior(deter, embed)
        post_sample = diag_normal(post_mean, post_std).rsample()
        obs_reconstr = self._decoder(post_sample)

        return (
            unflatten(obs_reconstr, n),  # tensor(N, B, C, H, W)
            unflatten(post_mean, n),     # tensor(N, B, S)
            unflatten(post_std, n),      # tensor(N, B, S)
        )

    def loss(self,
             obs_reconstr, post_mean, post_std,  # forward() output
             obs_target,                         # tensor(N, B, C, H, W)
             ):
        prior_mean = torch.zeros_like(post_mean)
        prior_std = torch.ones_like(post_std)
        loss_kl = D.kl.kl_divergence(diag_normal(post_mean, post_std), diag_normal(prior_mean, prior_std))
        loss_obs = self._decoder.loss(obs_reconstr, obs_target)
        assert loss_kl.shape == loss_obs.shape  # Should be (N, B)

        loss_kl = loss_kl.mean()  # Mean over (N, B)
        loss_obs = loss_obs.mean()
        loss = loss_kl + loss_obs
        metrics = dict(loss_kl=loss_kl.detach(), loss_obs=loss_obs.detach())
        return loss, metrics
