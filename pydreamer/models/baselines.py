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


class DreamerProbeVAE(nn.Module):

    def __init__(self, conf):
        super().__init__()

        # World model

        self.wm = VAEWorldModel(conf)

        # Map probe

        if conf.probe_model == 'map':
            probe_model = MapProbeHead(self.wm.features_dim + 4, conf)
        elif conf.probe_model == 'goals':
            probe_model = GoalsProbe(self.wm.features_dim, conf)
        elif conf.probe_model == 'none':
            probe_model = NoProbeHead()
        else:
            raise NotImplementedError(f'Unknown probe_model={conf.probe_model}')
        self.probe_model = probe_model

    def init_optimizers(self, lr, lr_actor=None, lr_critic=None, eps=1e-5):
        optimizer_wm = torch.optim.AdamW(self.wm.parameters(), lr=lr, eps=eps)
        optimizer_probe = torch.optim.AdamW(self.probe_model.parameters(), lr=lr, eps=eps)
        return optimizer_wm, optimizer_probe

    def grad_clip(self, grad_clip, grad_clip_ac=None):
        grad_metrics = {
            'grad_norm': nn.utils.clip_grad_norm_(self.wm.parameters(), grad_clip),
            'grad_norm_probe': nn.utils.clip_grad_norm_(self.probe_model.parameters(), grad_clip),
        }
        return grad_metrics

    def init_state(self, batch_size: int):
        return self.wm.init_state(batch_size)

    def training_step(self,
                      obs: Dict[str, Tensor],
                      in_state: Any,
                      iwae_samples: int = 1,
                      imag_horizon: int = None,
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

        loss_probe, metrics_probe, tensors_probe = self.probe_model.training_step(features.detach(), obs)
        metrics.update(**metrics_probe)
        tensors.update(**tensors_probe)

        return (loss_model, loss_probe), out_state, metrics, tensors, {}


class VAEWorldModel(nn.Module):

    def __init__(self, conf):
        super().__init__()
        self.kl_weight = conf.kl_weight
        self.features_dim = conf.stoch_dim
        self.encoder = MultiEncoder(conf)
        self.post_mlp = nn.Sequential(nn.Linear(self.encoder.out_dim, 256),
                                      nn.ELU(),
                                      nn.Linear(256, 2 * conf.stoch_dim))
        self.decoder = MultiDecoder(conf.stoch_dim, conf)
        for m in self.modules():
            init_weights_tf2(m)

    def init_state(self, batch_size: int) -> Any:
        return None

    def training_step(self,
                      obs: Dict[str, Tensor],
                      in_state: Any,
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
