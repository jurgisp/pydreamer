import torch
import torch.nn as nn
import torch.distributions as D

from modules import *


class RSSM(nn.Module):

    def __init__(self, encoder, decoder, map_model, deter_dim=200, stoch_dim=30, hidden_dim=200):
        super().__init__()
        self._encoder = encoder
        self._decoder_image = decoder
        self._map_model = map_model
        self._deter_dim = deter_dim
        self._stoch_dim = stoch_dim
        self._core = RSSMCore(embed_dim=encoder.out_dim,
                              deter_dim=deter_dim,
                              stoch_dim=stoch_dim,
                              hidden_dim=hidden_dim)
        for m in self.modules():
            init_weights_tf2(m)

    def forward(self,
                image,     # tensor(N, B, C, H, W)
                action,    # tensor(N, B, A)
                reset,     # tensor(N, B)
                map,       # tensor(N, B, C, MH, MW)
                in_state,  # tensor(   B, D+S)
                ):

        n = image.size(0)
        embed = unflatten(self._encoder(flatten(image)), n)
        prior, post, states = self._core(embed, action, reset, in_state)
        states_flat = flatten(states)
        image_rec = unflatten(self._decoder_image(states_flat), n)
        map_out = self._map_model(map, states.detach())

        return (
            prior,                       # tensor(N, B, 2*S)
            post,                        # tensor(N, B, 2*S)
            image_rec,                   # tensor(N, B, C, H, W)
            map_out,                     # tuple, map.forward() output
            states,                      # tensor(N, B, D+S)
        )

    def init_state(self, batch_size):
        return self._core.init_state(batch_size)

    def loss(self,
             prior, post, image_rec, map_out, states,     # forward() output
             image,                                      # tensor(N, B, C, H, W)
             map,                                        # tensor(N, B, MH, MW)
             ):
        loss_kl = D.kl.kl_divergence(diag_normal(post), diag_normal(prior))
        loss_image = self._decoder_image.loss(image_rec, image)
        log_tensors = dict(loss_kl=loss_kl.detach())
        assert loss_kl.shape == loss_image.shape
        loss_kl = loss_kl.mean()        # (N, B) => ()
        loss_image = loss_image.mean()
        loss = loss_kl + loss_image

        loss_map, metrics_map = self._map_model.loss(*map_out, map)
        metrics_map = {k.replace('loss_', 'loss_map_'): v for k, v in metrics_map.items()}  # loss_kl => loss_map_kl
        loss += loss_map

        metrics = dict(loss_kl=loss_kl.detach(),
                       loss_image=loss_image.detach(),
                       loss_model=loss_kl.detach() + loss_image.detach(),  # model loss, without detached heads
                       loss_map=loss_map.detach(),
                       **metrics_map)
        return loss, metrics, log_tensors

    def predict_obs(self,
                    prior, post, image_rec, map_out, states,  # forward() output
                    ):
        n = states.size(0)

        # Make states with z sampled from prior instead of posterior
        h, z_post = split(states, [self._deter_dim, self._stoch_dim])
        z_prior = diag_normal(prior).sample()
        states_prior = cat(h, z_prior)
        image_pred = unflatten(self._decoder_image(flatten(states_prior)), n)  # (N,B,C,H,W)

        image_pred_distr = D.Categorical(logits=image_pred.permute(0, 1, 3, 4, 2))  # (N,B,C,H,W) => (N,B,H,W,C)
        image_rec_distr = D.Categorical(logits=image_rec.permute(0, 1, 3, 4, 2))
        map_rec_distr = self._map_model.predict_obs(*map_out)

        return (
            image_pred_distr,    # categorical(N,B,H,W,C)
            image_rec_distr,     # categorical(N,B,H,W,C)
            map_rec_distr,       # categorical(N,B,HM,WM,C)
        )


class CondVAE(nn.Module):
    # Conditioned VAE

    def __init__(self, encoder, decoder, state_dim=230, hidden_dim=200, latent_dim=30):
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder
        assert decoder.in_dim == state_dim + latent_dim

        self._prior_mlp = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                                        nn.ELU(),
                                        nn.Linear(hidden_dim, 2 * latent_dim))

        self._post_mlp = nn.Sequential(nn.Linear(state_dim + encoder.out_dim, hidden_dim),
                                       nn.ELU(),
                                       nn.Linear(hidden_dim, 2 * latent_dim))

        self._min_std = 0.1

    def forward(self,
                obs,       # tensor(N, B, C, H, W)
                state,     # tensor(N, B, D+S)
                ):
        states_in = state

        n = obs.size(0)
        embed = self._encoder(flatten(obs))
        state = flatten(state)

        prior = to_mean_std(self._prior_mlp(state), self._min_std)              # (N*B, 2*Z)
        post = to_mean_std(self._post_mlp(cat(state, embed)), self._min_std)    # (N*B, 2*Z)
        sample = diag_normal(post).rsample()                                    # (N*B, Z)
        obs_rec = self._decoder(cat(state, sample))

        return (
            unflatten(prior, n),         # tensor(N, B, 2*Z)
            unflatten(post, n),          # tensor(N, B, 2*Z)
            unflatten(obs_rec, n),       # tensor(N, B, C, H, W)
            states_in
        )

    def loss(self,
             prior, post, obs_rec, states,       # forward() output
             obs_target,                         # tensor(N, B, C, H, W)
             ):
        loss_kl = D.kl.kl_divergence(diag_normal(post), diag_normal(prior))
        loss_rec = self._decoder.loss(obs_rec, obs_target)
        assert loss_kl.shape == loss_rec.shape
        loss_kl, loss_rec = loss_kl.mean(), loss_rec.mean()  # (N, B) => ()
        loss = loss_kl + loss_rec
        metrics = dict(loss_kl=loss_kl.detach(), loss_rec=loss_rec.detach())
        return loss, metrics

    def predict_obs(self,
                    prior, post, obs_rec, states,                 # forward() output
                    ):
        n = prior.size(0)
        # Sample from prior instead of posterior
        sample = diag_normal(prior).sample()
        obs_pred = unflatten(self._decoder(flatten(cat(states, sample))), n)    # (N,B,C,MH,MW)
        obs_pred_distr = D.Categorical(logits=obs_pred.permute(0, 1, 3, 4, 2))  # (N,B,C,MH,MW) => (N,B,MH,MW,C)
        return obs_pred_distr       # categorical(N,B,HM,WM,C)


class DirectHead(nn.Module):

    def __init__(self, decoder):
        super().__init__()
        self._decoder = decoder

    def forward(self,
                obs,       # tensor(N, B, C, H, W)
                state,     # tensor(N, B, D+S)
                ):
        n = obs.size(0)
        obs_pred = self._decoder(flatten(state))
        return (
            unflatten(obs_pred, n),       # tensor(N, B, C, H, W)
        )

    def loss(self,
             obs_pred,          # forward() output
             obs_target,        # tensor(N, B, C, H, W)
             ):
        loss = self._decoder.loss(obs_pred, obs_target).mean()
        return loss, {}

    def predict_obs(self,
                    obs_pred,                 # forward() output
                    ):
        obs_pred_distr = D.Categorical(logits=obs_pred.permute(0, 1, 3, 4, 2))  # (N,B,C,MH,MW) => (N,B,MH,MW,C)
        return obs_pred_distr       # categorical(N,B,HM,WM,C)
