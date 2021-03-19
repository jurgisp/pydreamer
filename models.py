import torch
import torch.nn as nn
import torch.distributions as D

from modules import *


class RSSM(nn.Module):

    def __init__(self, encoder, decoder, deter_dim=200, stoch_dim=30, hidden_dim=200):
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._deter_dim = deter_dim
        self._stoch_dim = stoch_dim
        self._core = RSSMCore(embed_dim=encoder.out_dim,
                              deter_dim=deter_dim,
                              stoch_dim=stoch_dim,
                              hidden_dim=hidden_dim)
        for m in self.modules():
            init_weights_tf2(m)

    def forward(self,
                obs,       # tensor(N, B, C, H, W)
                action,    # tensor(N, B, A)
                reset,     # tensor(N, B)
                in_state,  # tensor(   B, D+S)
                ):

        n = obs.size(0)
        embed = unflatten(self._encoder(flatten(obs)), n)
        prior, post, states = self._core(embed, action, reset, in_state)
        obs_reconstr = unflatten(self._decoder(flatten(states)), n)

        return (
            prior,                       # tensor(N, B, 2*S)
            post,                        # tensor(N, B, 2*S)
            obs_reconstr,                # tensor(N, B, C, H, W)
            states,                      # tensor(N, B, D+S)
        )

    def init_state(self, batch_size):
        return self._core.init_state(batch_size)

    def loss(self,
             prior, post, obs_reconstr, states,     # forward() output
             obs_target,                            # tensor(N, B, C, H, W)
             ):
        loss_kl = D.kl.kl_divergence(diag_normal(post), diag_normal(prior))
        loss_image = self._decoder.loss(obs_reconstr, obs_target)
        assert loss_kl.shape == loss_image.shape  # Should be (N, B)

        loss_kl_mean = loss_kl.mean()  # Mean over (N, B)
        loss_image_mean = loss_image.mean()
        loss = loss_kl_mean + loss_image_mean

        metrics = dict(loss_kl=loss_kl_mean.detach(), loss_image=loss_image_mean.detach())
        tensors = dict(loss_kl=loss_kl.detach(), loss_image=loss_image.detach())
        return loss, metrics, tensors

    def predict_obs(self,
                    prior, post, obs_reconstr, states,  # forward() output
                    ):
        n = states.size(0)

        # Make states with z sampled from prior instead of posterior
        h, z_post = split(states, [self._deter_dim, self._stoch_dim])
        z_prior = diag_normal(prior).sample()
        states_prior = cat(h, z_prior)
        obs_pred = unflatten(self._decoder(flatten(states_prior)), n)  # (N, B, C, H, W)

        obs_pred_sample = D.Categorical(logits=obs_pred.permute(0, 1, 3, 4, 2)).sample()
        obs_reconstr_sample = D.Categorical(logits=obs_reconstr.permute(0, 1, 3, 4, 2)).sample()
        return (
            obs_pred_sample,     # tensor(N, B, H, W)
            obs_reconstr_sample  # tensor(N, B, H, W)
        )


class VAE(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._post_mlp = nn.Sequential(nn.Linear(encoder.out_dim, 256),
                                       nn.ELU(),
                                       nn.Linear(256, 2 * decoder.in_dim))
        self._min_std = 0.1

    def init_state(self, batch_size):
        return torch.zeros(batch_size)

    def forward(self,
                obs,       # tensor(N, B, C, H, W)
                action,    # tensor(N, B, A)
                reset,     # tensor(N, B)
                in_state,  # tensor(   B, D+S)
                ):

        n = obs.size(0)
        embed = self._encoder(flatten(obs))
        post = to_mean_std(self._post_mlp(embed), self._min_std)
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
        loss_image = self._decoder.loss(obs_reconstr, obs_target)
        assert loss_kl.shape == loss_image.shape  # Should be (N, B)

        loss_kl = loss_kl.mean()  # Mean over (N, B)
        loss_image = loss_image.mean()
        loss = loss_kl + loss_image
        metrics = dict(loss_kl=loss_kl.detach(), loss_image=loss_image.detach())
        return loss, metrics
