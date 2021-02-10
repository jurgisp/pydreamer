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

        n = obs.size(0)
        embed = self._encoder(flatten(obs))

        deter = torch.zeros_like(embed)
        post = self._posterior(deter, embed)
        post_sample = diag_normal(post).rsample()
        obs_reconstr = self._decoder(post_sample)

        return (
            unflatten(obs_reconstr, n),  # tensor(N, B, C, H, W)
            unflatten(post, n),          # tensor(N, B, S+S)
        )

    def loss(self,
             obs_reconstr, post,                 # forward() output
             obs_target,                         # tensor(N, B, C, H, W)
             ):
        prior = zero_prior_like(post)
        loss_kl = D.kl.kl_divergence(diag_normal(post), diag_normal(prior))
        loss_obs = self._decoder.loss(obs_reconstr, obs_target)
        assert loss_kl.shape == loss_obs.shape  # Should be (N, B)

        loss_kl = loss_kl.mean()  # Mean over (N, B)
        loss_obs = loss_obs.mean()
        loss = loss_kl + loss_obs
        metrics = dict(loss_kl=loss_kl.detach(), loss_obs=loss_obs.detach())
        return loss, metrics
