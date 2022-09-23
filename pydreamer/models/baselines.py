from typing import Any, Tuple

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from ..tools import *
from .a2c import *
from .common import *
from .decoders import *
from .encoders import *
from .functions import *
from .probes import *
from .rnn import *
from .rssm import *


class WorldModelProbe(nn.Module):

    def __init__(self, conf):
        super().__init__()
        self.probe_gradients = conf.probe_gradients

        # World model

        if conf.model == 'vae':
            self.wm = VAEWorldModel(conf)
        elif conf.model == 'gru_vae':
            self.wm = GRUVAEWorldModel(conf)
        elif conf.model == 'transformer_vae':
            self.wm = TransformerVAEWorldModel(conf)
        elif conf.model == 'gru_probe':
            self.wm = GRUEncoderOnly(conf)
        else:
            raise ValueError(conf.model)

        # Map probe

        if conf.probe_model == 'map':
            probe_model = MapProbeHead(self.wm.out_dim + 4, conf)
        elif conf.probe_model == 'goals':
            probe_model = GoalsProbe(self.wm.out_dim, conf)
        elif conf.probe_model == 'map+goals':
            probe_model = MapGoalsProbe(self.wm.out_dim, conf)
        elif conf.probe_model == 'none':
            probe_model = NoProbeHead()
        else:
            raise NotImplementedError(f'Unknown probe_model={conf.probe_model}')
        self.probe_model = probe_model

        # Use TF2 weight initialization (TODO: Dreamer is missing this on probe model)

        for m in self.modules():
            init_weights_tf2(m)

    def init_optimizers(self, lr, lr_actor=None, lr_critic=None, eps=1e-5):
        if not self.probe_gradients:
            optimizer_wm = torch.optim.AdamW(self.wm.parameters(), lr=lr, eps=eps)
            optimizer_probe = torch.optim.AdamW(self.probe_model.parameters(), lr=lr, eps=eps)
            return optimizer_wm, optimizer_probe
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, eps=eps)
            return (optimizer,)

    def grad_clip(self, grad_clip, grad_clip_ac=None):
        if not self.probe_gradients:
            grad_metrics = {
                'grad_norm': nn.utils.clip_grad_norm_(self.wm.parameters(), grad_clip),
                'grad_norm_probe': nn.utils.clip_grad_norm_(self.probe_model.parameters(), grad_clip),
            }
        else:
            grad_metrics = {
                'grad_norm': nn.utils.clip_grad_norm_(self.parameters(), grad_clip),
            }
        return grad_metrics

    def init_state(self, batch_size: int):
        return self.wm.init_state(batch_size)

    def training_step(self,
                      obs: Dict[str, Tensor],
                      in_state: Any,
                      iwae_samples: int = 1,
                      imag_horizon: Optional[int] = None,
                      do_open_loop=False,
                      do_image_pred=False,
                      do_dream_tensors=False,
                      ):
        # World model

        loss_model, features, states, out_state, metrics, tensors = \
            self.wm.training_step(obs,
                                  in_state,
                                  iwae_samples=iwae_samples,
                                  do_open_loop=do_open_loop,
                                  do_image_pred=do_image_pred)

        # Probe

        if not self.probe_gradients:
            features = features.detach()
        loss_probe, metrics_probe, tensors_probe = self.probe_model.training_step(features, obs)
        metrics.update(**metrics_probe)
        tensors.update(**tensors_probe)

        if not self.probe_gradients:
            losses = (loss_model, loss_probe)
        else:
            losses = (loss_model + loss_probe,)
        return losses, out_state, metrics, tensors, {}


class GRUVAEWorldModel(nn.Module):

    def __init__(self, conf):
        super().__init__()
        self.state_dim = conf.deter_dim
        self.out_dim = self.state_dim
        self.embedding = VAEWorldModel(conf)
        self.rnn = nn.GRU(self.embedding.out_dim + conf.action_dim, self.state_dim)
        self.dynamics = DenseNormalDecoder(self.state_dim, self.embedding.out_dim, hidden_layers=2)

    def init_state(self, batch_size: int) -> Any:
        device = next(self.rnn.parameters()).device
        return torch.zeros((1, batch_size, self.state_dim), device=device)

    def training_step(self,
                      obs: Dict[str, Tensor],
                      in_state: Any,
                      iwae_samples: int = 1,
                      do_open_loop=False,
                      do_image_pred=False,
                      ):

        # Reset state if needed

        reset_first = obs['reset'].select(0, 0)  # Assume only reset on batch start (allow_mid_reset=False)
        state_mask = ~reset_first.unsqueeze(0).unsqueeze(-1)
        in_state = in_state * state_mask

        # VAE embedding

        loss, embed, _, _, metrics, tensors = \
            self.embedding.training_step(obs, None,
                                         iwae_samples=iwae_samples,
                                         do_image_pred=do_image_pred)
        T, B, I = embed.shape[:3]
        embed = embed.reshape((T, B * I, -1))  # (T,B,I,*) => (T,BI,*)
        embed = embed.detach()  # Predict embeddings as they are

        # Embedding + action

        action_next = obs['action_next']
        embed_act = torch.cat([embed, action_next], -1)

        # RNN

        features, out_state = self.rnn(embed_act, in_state)
        features = features.reshape((T, B, I, -1))  # (T,BI,*) => (T,B,I,*)
        out_state = out_state.detach()  # Detach before next batch

        # Predict

        embed_next = embed[1:]
        _, loss_dyn, embed_pred = self.dynamics.training_step(features[:-1], embed_next)
        loss += loss_dyn.mean()
        metrics['loss_dyn'] = loss_dyn.detach().mean()
        tensors['loss_dyn'] = loss_dyn.detach()

        if do_image_pred:
            with torch.no_grad():
                # Decode from embed prediction
                z = embed_pred
                z = torch.cat([torch.zeros_like(z[0]).unsqueeze(0), z])  # we don't have prediction for first step, so insert zeros
                _, mets, tens = self.embedding.decoder.training_step(z.unsqueeze(2), obs, extra_metrics=True)
                tensors_pred = {k.replace('_rec', '_pred'): v for k, v in tens.items() if k.endswith('_rec')}
                tensors.update(**tensors_pred)  # image_pred, ..

        return loss, features, None, out_state, metrics, tensors


class TransformerVAEWorldModel(nn.Module):

    def __init__(self, conf):
        super().__init__()
        self.state_dim = 512
        self.out_dim = self.state_dim
        self.embedding = VAEWorldModel(conf)
        self.transformer_in = nn.Linear(self.embedding.out_dim + conf.action_dim, 512)
        self.transformer = nn.TransformerEncoder(
            # defaults
            nn.TransformerEncoderLayer(512, nhead=8, dim_feedforward=2048, dropout=0.1),
            num_layers=6,
            norm=nn.LayerNorm(512)
        )
        self.dynamics = DenseNormalDecoder(self.state_dim, self.embedding.out_dim, hidden_layers=2)

    def init_state(self, batch_size: int) -> Any:
        return None

    def training_step(self,
                      obs: Dict[str, Tensor],
                      in_state: Any,
                      iwae_samples: int = 1,
                      do_open_loop=False,
                      do_image_pred=False,
                      ):
        # VAE embedding

        loss, embed, _, _, metrics, tensors = \
            self.embedding.training_step(obs, None,
                                         iwae_samples=iwae_samples,
                                         do_image_pred=do_image_pred)
        T, B, I = embed.shape[:3]
        embed = embed.reshape((T, B * I, -1))  # (T,B,I,*) => (T,BI,*)
        embed = embed.detach()  # Predict embeddings as they are
        action_next = obs['action_next']
        embed_act = torch.cat([embed, action_next], -1)

        # TRANSFORMER

        # TODO: maybe scale by  * math.sqrt(self.ninp)
        # TODO: positional encodding
        # TODO: masking
        state_in = self.transformer_in(embed_act)
        features = self.transformer(state_in)

        # Predict

        features = features.reshape((T, B, I, -1))  # (T,BI,*) => (T,B,I,*)
        embed_next = embed[1:]
        _, loss_dyn, embed_pred = self.dynamics.training_step(features[:-1], embed_next)
        loss += loss_dyn.mean()
        metrics['loss_dyn'] = loss_dyn.detach().mean()
        tensors['loss_dyn'] = loss_dyn.detach()

        if do_image_pred:
            with torch.no_grad():
                # Decode from embed prediction
                z = embed_pred
                z = torch.cat([torch.zeros_like(z[0]).unsqueeze(0), z])  # we don't have prediction for first step, so insert zeros
                _, mets, tens = self.embedding.decoder.training_step(z.unsqueeze(2), obs, extra_metrics=True)
                tensors_pred = {k.replace('_rec', '_pred'): v for k, v in tens.items() if k.endswith('_rec')}
                tensors.update(**tensors_pred)  # image_pred, ..

        return loss, features, None, None, metrics, tensors


class VAEWorldModel(nn.Module):

    def __init__(self, conf):
        super().__init__()
        self.kl_weight = conf.kl_weight
        self.out_dim = conf.stoch_dim
        self.encoder = MultiEncoder(conf)
        self.post_mlp = nn.Sequential(nn.Linear(self.encoder.out_dim, 256),
                                      nn.ELU(),
                                      nn.Linear(256, 2 * conf.stoch_dim))
        self.decoder = MultiDecoder(conf.stoch_dim, conf)

    def init_state(self, batch_size: int):
        return None

    def training_step(self,
                      obs: Dict[str, Tensor],
                      in_state: Any = None,
                      iwae_samples: int = 1,
                      do_open_loop=False,
                      do_image_pred=False,
                      ):
        # Encode-sample-decode

        embed = self.encoder(obs)
        post = self.post_mlp(embed)
        post = insert_dim(post, 2, iwae_samples)
        post_distr = diag_normal(post)
        z = post_distr.rsample()
        loss_reconstr, metrics, tensors = self.decoder.training_step(z, obs)

        # Loss

        prior_distr = diag_normal(torch.zeros_like(post))  # ~ Normal(0,1)
        loss_kl = D.kl.kl_divergence(post_distr, prior_distr)  # (T,B,I)
        assert loss_kl.shape == loss_reconstr.shape
        loss_model_tbi = self.kl_weight * loss_kl + loss_reconstr
        loss_model = -logavgexp(-loss_model_tbi, dim=2)

        # Metrics

        with torch.no_grad():
            loss_kl = -logavgexp(-loss_kl, dim=2)
            entropy_post = post_distr.entropy().mean(dim=2)
            tensors.update(loss_kl=loss_kl.detach(),
                           entropy_post=entropy_post)
            metrics.update(loss_model=loss_model.mean(),
                           loss_kl=loss_kl.mean(),
                           entropy_post=entropy_post.mean())

        # Predictions (from unconditioned prior)

        if do_image_pred:
            with torch.no_grad():
                # Decode from prior sample
                zprior = prior_distr.sample()
                _, mets, tens = self.decoder.training_step(zprior, obs, extra_metrics=True)
                tensors_pred = {k.replace('_rec', '_pred'): v for k, v in tens.items() if k.endswith('_rec')}
                tensors.update(**tensors_pred)  # image_pred, ...

        return loss_model.mean(), z, None, None, metrics, tensors



class GRUEncoderOnly(nn.Module):

    def __init__(self, conf):
        super().__init__()
        self.state_dim = conf.deter_dim
        self.out_dim = self.state_dim
        self.encoder = MultiEncoder(conf)
        self.squeeze = nn.Linear(self.encoder.out_dim, 32)
        self.rnn = nn.GRU(32 + conf.action_dim, self.state_dim)

    def init_state(self, batch_size: int) -> Any:
        device = next(self.rnn.parameters()).device
        return torch.zeros((1, batch_size, self.state_dim), device=device)

    def training_step(self,
                      obs: Dict[str, Tensor],
                      in_state: Any,
                      iwae_samples: int = 1,
                      do_open_loop=False,
                      do_image_pred=False,
                      ):
        assert iwae_samples == 1
        
        # Reset state if needed

        reset_first = obs['reset'].select(0, 0)  # Assume only reset on batch start (allow_mid_reset=False)
        state_mask = ~reset_first.unsqueeze(0).unsqueeze(-1)
        in_state = in_state * state_mask

        # Encoder

        embed = self.encoder(obs)
        embed = self.squeeze(embed)  # squeeze to smaller dimension so embedding input doesn't dominate action input
        action_next = obs['action_next']
        embed_act = torch.cat([embed, action_next], -1)

        # RNN

        features, out_state = self.rnn(embed_act, in_state)
        out_state = out_state.detach()  # Detach before next batch

        features = features.unsqueeze(-2)  # Insert I=1 dim
        loss = 0.0   # This is forward-only
        return loss, features, None, out_state, {}, {}