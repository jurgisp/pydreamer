from typing import Any, Tuple

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from tools import *
from models.a2c import *
from models.common import *
from models.functions import *
from models.encoders import *
from models.decoders import *
from models.rnn import *
from models.rssm import *
from models.probes import *


class Dreamer(nn.Module):

    def __init__(self, conf):
        super().__init__()

        state_dim = conf.deter_dim + conf.stoch_dim * (conf.stoch_discrete or 1)

        # World model

        self.wm = WorldModel(conf)

        # Actor critic

        self.ac = ActorCritic(in_dim=state_dim,
                              out_actions=conf.action_dim,
                              layer_norm=conf.layer_norm,
                              gamma=conf.gamma,
                              lambda_gae=conf.lambda_gae,
                              entropy_weight=conf.entropy,
                              target_interval=conf.target_interval,
                              )

        # Map probe

        if conf.map_model == 'direct':
            map_model = MapProbeHead(state_dim + 4, conf)
        elif conf.map_model == 'none':
            map_model = NoProbeHead()
        else:
            raise NotImplementedError(f'Unknown map_model={conf.map_model}')
        self.map_model = map_model

    def init_optimizers(self, conf):
        optimizer_wm = torch.optim.AdamW(self.wm.parameters(), lr=conf.adam_lr, eps=conf.adam_eps)  # type: ignore
        optimizer_map = torch.optim.AdamW(self.map_model.parameters(), lr=conf.adam_lr, eps=conf.adam_eps)  # type: ignore
        optimizer_actor = torch.optim.AdamW(self.ac.actor.parameters(), lr=conf.adam_lr_actor, eps=conf.adam_eps)  # type: ignore
        optimizer_critic = torch.optim.AdamW(self.ac.critic.parameters(), lr=conf.adam_lr_critic, eps=conf.adam_eps)  # type: ignore
        return optimizer_wm, optimizer_map, optimizer_actor, optimizer_critic

    def grad_clip(self, conf):
        grad_metrics = {
            'grad_norm': nn.utils.clip_grad_norm_(self.wm.parameters(), conf.grad_clip),
            'grad_norm_map': nn.utils.clip_grad_norm_(self.map_model.parameters(), conf.grad_clip),
            'grad_norm_actor': nn.utils.clip_grad_norm_(self.ac.actor.parameters(), conf.grad_clip_ac),
            'grad_norm_critic': nn.utils.clip_grad_norm_(self.ac.critic.parameters(), conf.grad_clip_ac),
        }
        return grad_metrics

    def init_state(self, batch_size: int):
        return self.wm.init_state(batch_size)

    def forward(self,
                obs: Dict[str, Tensor],
                in_state: Any,
                ):
        assert 'action' in obs, 'Observation should contain previous action'
        act_shape = obs['action'].shape
        assert len(act_shape) == 3 and act_shape[0] == 1, f'Expected shape (1,B,A), got {act_shape}'

        # Forward (world model)

        features, out_state = self.wm.forward(obs, in_state)

        # Forward (actor critic)

        feature = features[:, :, 0]  # (N=1,B,I=1,F) => (1,B,F)
        action_logits = self.ac.forward_actor(feature)  # (1,B,A)
        value = self.ac.forward_value(feature)  # (1,B)

        metrics = dict(policy_value=value.detach().mean())
        return action_logits, out_state, metrics

    def training_step(self,
                      obs: Dict[str, Tensor],
                      in_state: Any,
                      iwae_samples: int = 1,
                      imag_horizon: int = 1,
                      do_open_loop=False,
                      do_image_pred=False,
                      do_dream_tensors=False,
                      ):
        assert 'action' in obs, '`action` required in observation'
        assert 'reward' in obs, '`reward` required in observation'
        assert 'reset' in obs, '`reset` required in observation'
        assert 'terminal' in obs, '`terminal` required in observation'
        N, B = obs['action'].shape[:2]
        I, H = iwae_samples, imag_horizon

        # World model

        loss_model, features, states, out_state, metrics, tensors = \
            self.wm.training_step(obs,
                                  in_state,
                                  iwae_samples=iwae_samples,
                                  do_open_loop=do_open_loop,
                                  do_image_pred=do_image_pred)

        # Map probe

        loss_map, metrics_map, tensors_map = self.map_model.training_step(features.detach(), obs)
        metrics.update(**metrics_map)
        tensors.update(**tensors_map)

        # Policy

        in_state_dream: StateB = map_structure(states, lambda x: flatten_batch(x.detach())[0])  # type: ignore  # (N,B,I) => (NBI)
        features_dream, actions_dream = self.dream(in_state_dream, H)   # (H+1,NBI,D) - features_dream includes the starting "real" features at features_dream[0]
        features_dream = features_dream.detach()  # Not using dynamics gradients for now, just Reinforce
        with torch.no_grad():  # careful not to invoke modules first time under no_grad (https://github.com/pytorch/pytorch/issues/60164)
            rewards_dream = self.wm.decoder.reward.forward(features_dream)      # (H+1,NBI)
            terminals_dream = self.wm.decoder.terminal.forward(features_dream)  # (H+1,NBI)

        losses_ac, metrics_ac, tensors_ac = self.ac.training_step(features_dream, rewards_dream, terminals_dream, actions_dream)
        metrics.update(**metrics_ac)
        tensors.update(policy_value=unflatten_batch(tensors_ac['value'][0], (N, B, I)).mean(-1))

        # Dream for a log sample.

        dream_tensors = {}
        if do_dream_tensors:
            with torch.no_grad():  # careful not to invoke modules first time under no_grad (https://github.com/pytorch/pytorch/issues/60164)
                # The reason we don't just take real features_dream is because it's really big (H*N*B*I),
                # and here for inspection purposes we only dream from first step, so it's (H*B).
                # Oh, and we set here H=N-1, so we get (N,B), and the dreamed experience aligns with actual.
                in_state_dream: StateB = map_structure(states, lambda x: x.detach()[0, :, 0])  # type: ignore  # (N,B,I) => (B)
                features_dream, actions_dream = self.dream(in_state_dream, N - 1)      # H = N-1
                rewards_dream = self.wm.decoder.reward.forward(features_dream)      # (H+1,B) = (N,B)
                terminals_dream = self.wm.decoder.terminal.forward(features_dream)  # (H+1,B) = (N,B)
                image_dream = self.wm.decoder.image.forward(features_dream)
                _, _, tensors_ac = self.ac.training_step(features_dream, rewards_dream, terminals_dream, actions_dream, log_only=True)
                # The tensors are intentionally named same as in tensors, so the logged npz looks the same for dreamed or not
                dream_tensors = dict(action_pred=torch.cat([obs['action'][:1], actions_dream]),  # first action is real from previous step
                                     reward_pred=rewards_dream.mean,
                                     terminal_pred=terminals_dream.mean,
                                     image_pred=image_dream,
                                     **tensors_ac)
                assert dream_tensors['action_pred'].shape == obs['action'].shape
                assert dream_tensors['image_pred'].shape == obs['image'].shape

        return (loss_model, loss_map, *losses_ac), out_state, metrics, tensors, dream_tensors

    def dream(self, in_state: StateB, imag_horizon: int):
        NBI = len(in_state[0])  # Imagine batch size = N*B*I
        H = imag_horizon
        features = []
        actions = []
        state = in_state

        for i in range(imag_horizon):
            feature = self.wm.core.to_feature(*state)
            action = D.OneHotCategorical(logits=self.ac.forward_actor(feature)).sample()
            features.append(feature)
            actions.append(action)
            _, state = self.wm.core.cell.forward_prior(action, None, state)

        feature = self.wm.core.to_feature(*state)
        features.append(feature)

        features = torch.stack(features)  # (H+1,NBI,D)
        actions = torch.stack(actions)  # (H,NBI,A)
        return features, actions

    def __str__(self):
        # Short representation
        s = []
        s.append(f'Model: {param_count(self)} parameters')
        for submodel in (self.wm.encoder, self.wm.decoder, self.wm.core, self.ac, self.map_model):
            if submodel is not None:
                s.append(f'  {type(submodel).__name__:<15}: {param_count(submodel)} parameters')
        return '\n'.join(s)

    def __repr__(self):
        # Long representation
        return super().__repr__()


class WorldModel(nn.Module):

    def __init__(self, conf):
        super().__init__()

        self.deter_dim = conf.deter_dim
        self.stoch_dim = conf.stoch_dim
        self.stoch_discrete = conf.stoch_discrete
        self.kl_weight = conf.kl_weight
        self.kl_balance = None if conf.kl_balance == 0.5 else conf.kl_balance

        # Encoder

        self.encoder = MultiEncoder(conf)

        # Decoders

        features_dim = conf.deter_dim + conf.stoch_dim * (conf.stoch_discrete or 1)
        self.decoder = MultiDecoder(features_dim, conf)

        # RSSM

        self.core = RSSMCore(embed_dim=self.encoder.out_dim,
                             action_dim=conf.action_dim,
                             deter_dim=conf.deter_dim,
                             stoch_dim=conf.stoch_dim,
                             stoch_discrete=conf.stoch_discrete,
                             hidden_dim=conf.hidden_dim,
                             gru_layers=conf.gru_layers,
                             gru_type=conf.gru_type,
                             layer_norm=conf.layer_norm)

        # Init

        for m in self.modules():
            init_weights_tf2(m)

    def init_state(self, batch_size: int) -> Tuple[Any, Any]:
        return self.core.init_state(batch_size)

    def forward(self,
                obs: Dict[str, Tensor],
                in_state: Any
                ):
        loss, features, states, out_state, metrics, tensors = \
            self.training_step(obs, in_state, forward_only=True)
        return features, out_state

    def training_step(self,
                      obs: Dict[str, Tensor],
                      in_state: Any,
                      iwae_samples: int = 1,
                      do_open_loop=False,
                      do_image_pred=False,
                      forward_only=False
                      ):

        # Encoder

        embed = self.encoder(obs)

        # RSSM

        prior, post, post_samples, features, states, out_state = \
            self.core.forward(embed,
                              obs['action'],
                              obs['reset'],
                              in_state,
                              iwae_samples=iwae_samples,
                              do_open_loop=do_open_loop)

        if forward_only:
            return torch.tensor(0.0), features, states, out_state, {}, {}

        # Decoder

        loss_reconstr, metrics, tensors = self.decoder.training_step(features, obs)

        # KL loss

        d = self.core.zdistr
        dprior = d(prior)
        dpost = d(post)
        loss_kl_exact = D.kl.kl_divergence(dpost, dprior)  # (N,B,I)
        if iwae_samples == 1:
            # Analytic KL loss, standard for VAE
            if not self.kl_balance:
                loss_kl = loss_kl_exact
            else:
                loss_kl_postgrad = D.kl.kl_divergence(dpost, d(prior.detach()))
                loss_kl_priograd = D.kl.kl_divergence(d(post.detach()), dprior)
                loss_kl = (1 - self.kl_balance) * loss_kl_postgrad + self.kl_balance * loss_kl_priograd
        else:
            # Sampled KL loss, for IWAE
            z = post_samples.reshape(dpost.batch_shape + dpost.event_shape)
            loss_kl = dpost.log_prob(z) - dprior.log_prob(z)

        # Total loss

        assert loss_kl.shape == loss_reconstr.shape
        loss_model_nbi = self.kl_weight * loss_kl + loss_reconstr
        loss_model = -logavgexp(-loss_model_nbi, dim=2)

        # Metrics

        with torch.no_grad():
            loss_kl = -logavgexp(-loss_kl_exact, dim=2)  # Log exact KL loss even when using IWAE, it avoids random negative values
            entropy_prior = dprior.entropy().mean(dim=2)
            entropy_post = dpost.entropy().mean(dim=2)
            tensors.update(loss_kl=loss_kl.detach(),
                           entropy_prior=entropy_prior,
                           entropy_post=entropy_post)
            metrics.update(loss_model=loss_model.mean(),
                           loss_kl=loss_kl.mean(),
                           entropy_prior=entropy_prior.mean(),
                           entropy_post=entropy_post.mean())

        # Predictions

        if do_image_pred:
            with torch.no_grad():
                prior_samples = self.core.zdistr(prior).sample().reshape(post_samples.shape)
                features_prior = self.core.feature_replace_z(features, prior_samples)
                # Decode from prior
                _, mets, tens = self.decoder.training_step(features_prior, obs, extra_metrics=True)
                metrics_logprob = {k.replace('loss_', 'logprob_'): v for k, v in mets.items() if k.startswith('loss_')}
                tensors_logprob = {k.replace('loss_', 'logprob_'): v for k, v in tens.items() if k.startswith('loss_')}
                tensors_pred = {k.replace('_rec', '_pred'): v for k, v in tens.items() if k.endswith('_rec')}
                metrics.update(**metrics_logprob)   # logprob_image, ...
                tensors.update(**tensors_logprob)  # logprob_image, ...
                tensors.update(**tensors_pred)  # image_pred, ...

        return loss_model.mean(), features, states, out_state, metrics, tensors
