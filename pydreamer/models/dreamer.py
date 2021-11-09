from typing import Any, Tuple

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from tools import *
from models.a2c import *
from models.common import *
from models.functions import *
from models.io import *
from models.rnn import *
from models.rssm import *
from models.probes import *


class Dreamer(nn.Module):

    def __init__(self, conf):
        super().__init__()

        state_dim = conf.deter_dim + conf.stoch_dim * (conf.stoch_discrete or 1)

        # World model

        self.wm = WorldModel(
            conf,
            action_dim=conf.action_dim,
            deter_dim=conf.deter_dim,
            stoch_dim=conf.stoch_dim,
            stoch_discrete=conf.stoch_discrete,
            hidden_dim=conf.hidden_dim,
            kl_weight=conf.kl_weight,
            image_weight=conf.image_weight,
            vecobs_weight=conf.vecobs_weight,
            reward_weight=conf.reward_weight,
            terminal_weight=conf.terminal_weight,
            gru_layers=conf.gru_layers,
            gru_type=conf.gru_type,
            kl_balance=conf.kl_balance,
        )

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
        metrics.update(loss=metrics['loss_wm'])

        # Map probe

        loss_map, metrics_map, tensors_map = self.map_model.training_step(features.detach(), obs)
        metrics.update(**metrics_map)
        tensors.update(**tensors_map)

        # Policy

        in_state_dream: StateB = map_structure(states, lambda x: flatten_batch(x.detach())[0])  # type: ignore  # (N,B,I) => (NBI)
        features_dream, actions_dream = self.dream(in_state_dream, H)   # (H+1,NBI,D) - features_dream includes the starting "real" features at features_dream[0]
        features_dream = features_dream.detach()  # Not using dynamics gradients for now, just Reinforce
        with torch.no_grad():  # careful not to invoke modules first time under no_grad (https://github.com/pytorch/pytorch/issues/60164)
            rewards_dream = self.wm.decoder_reward.forward(features_dream)      # (H+1,NBI)
            terminals_dream = self.wm.decoder_terminal.forward(features_dream)  # (H+1,NBI)

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
                rewards_dream = self.wm.decoder_reward.forward(features_dream)      # (H+1,B) = (N,B)
                terminals_dream = self.wm.decoder_terminal.forward(features_dream)  # (H+1,B) = (N,B)
                image_dream = self.wm.decoder_image.forward(features_dream)
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
        for submodel in (self.wm.encoder, self.wm.decoder_image, self.wm.core, self.ac, self.map_model):
            if submodel is not None:
                s.append(f'  {type(submodel).__name__:<15}: {param_count(submodel)} parameters')
        return '\n'.join(s)

    def __repr__(self):
        # Long representation
        return super().__repr__()


class WorldModel(nn.Module):

    def __init__(self,
                 conf,
                 action_dim=7,
                 deter_dim=200,
                 stoch_dim=30,
                 stoch_discrete=0,
                 hidden_dim=200,
                 kl_weight=1.0,
                 image_weight=1.0,
                 vecobs_weight=1.0,
                 reward_weight=1.0,
                 terminal_weight=1.0,
                 gru_layers=1,
                 gru_type='gru',
                 kl_balance=0.5,
                 ):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.stoch_discrete = stoch_discrete
        self.kl_weight = kl_weight
        self.image_weight = image_weight
        self.vecobs_weight = vecobs_weight
        self.reward_weight = reward_weight
        self.terminal_weight = terminal_weight
        self.kl_balance = None if kl_balance == 0.5 else kl_balance

        # Encoder

        self.encoder = MultiEncoder(conf)

        # Decoders

        state_dim = conf.deter_dim + conf.stoch_dim * (conf.stoch_discrete or 1)
        if conf.image_decoder == 'cnn':
            self.decoder_image = ConvDecoder(in_dim=state_dim,
                                             out_channels=conf.image_channels,
                                             cnn_depth=conf.cnn_depth)
        else:
            self.decoder_image = DenseDecoder(in_dim=state_dim,
                                              out_shape=(conf.image_channels, conf.image_size, conf.image_size),
                                              hidden_layers=conf.image_decoder_layers,
                                              layer_norm=conf.layer_norm,
                                              min_prob=conf.image_decoder_min_prob)

        if conf.reward_decoder_categorical:
            self.decoder_reward = DenseCategoricalSupportHead(in_dim=state_dim,
                                                              support=conf.reward_decoder_categorical,
                                                              hidden_layers=conf.reward_decoder_layers,
                                                              layer_norm=conf.layer_norm)
        else:
            self.decoder_reward = DenseNormalHead(in_dim=state_dim, hidden_layers=conf.reward_decoder_layers, layer_norm=conf.layer_norm)
        self.decoder_terminal = DenseBernoulliHead(in_dim=state_dim, hidden_layers=conf.terminal_decoder_layers, layer_norm=conf.layer_norm)
        self.decoder_vecobs = DenseNormalHead(in_dim=state_dim, out_dim=64, hidden_layers=4, layer_norm=conf.layer_norm)

        # RSSM

        self.core = RSSMCore(embed_dim=self.encoder.out_dim,
                             action_dim=action_dim,
                             deter_dim=deter_dim,
                             stoch_dim=stoch_dim,
                             stoch_discrete=stoch_discrete,
                             hidden_dim=hidden_dim,
                             gru_layers=gru_layers,
                             gru_type=gru_type,
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

        image_rec = self.decoder_image.forward(features)
        vecobs_rec = self.decoder_vecobs.forward(features)
        reward_rec = self.decoder_reward.forward(features)
        terminal_rec = self.decoder_terminal.forward(features)

        # Predictions

        image_pred, reward_pred, terminal_pred = None, None, None
        if do_image_pred:
            prior_samples = self.core.zdistr(prior).sample().reshape(post_samples.shape)
            features_prior = self.core.feature_replace_z(features, prior_samples)
            image_pred = self.decoder_image.forward(features_prior)
            reward_pred = self.decoder_reward.forward(features_prior)
            terminal_pred = self.decoder_terminal.forward(features_prior)

        # ------ LOSS -------

        I = iwae_samples
        image = insert_dim(obs['image'], 2, I)  # TODO: do this expansion inside decoder.loss?
        vecobs = insert_dim(obs['vecobs'], 2, I)
        reward = insert_dim(obs['reward'], 2, I)
        terminal = insert_dim(obs['terminal'], 2, I)

        # Decoder loss

        loss_image = self.decoder_image.loss(image_rec, image)  # (N,B,I)
        loss_vecobs = self.decoder_vecobs.loss(vecobs_rec, vecobs)  # (N,B,I)
        loss_reward = self.decoder_reward.loss(reward_rec, reward)  # (N,B,I)
        loss_terminal = self.decoder_terminal.loss(terminal_rec, terminal)  # (N,B,I)
        assert (loss_image.requires_grad and loss_reward.requires_grad and loss_terminal.requires_grad) or not features.requires_grad

        # KL loss

        d = self.core.zdistr
        dprior = d(prior)
        dpost = d(post)
        loss_kl_exact = D.kl.kl_divergence(dpost, dprior)  # (N,B,I)
        if I == 1:
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

        assert loss_kl.shape == loss_image.shape == loss_vecobs.shape == loss_reward.shape == loss_terminal.shape
        loss_model = -logavgexp(-(
            self.kl_weight * loss_kl
            + self.image_weight * loss_image
            + self.vecobs_weight * loss_vecobs
            + self.reward_weight * loss_reward
            + self.terminal_weight * loss_terminal
        ), dim=-1)

        # IWAE according to paper

        # with torch.no_grad():
        #     weights = F.softmax(-(loss_image + loss_kl), dim=-1)
        # dloss = (weights * (loss_image + loss_kl)).sum(dim=-1).mean()

        # Metrics

        with torch.no_grad():
            loss_image = -logavgexp(-loss_image, dim=-1)
            loss_vecobs = -logavgexp(-loss_vecobs, dim=-1)
            loss_reward = -logavgexp(-loss_reward, dim=-1)
            loss_terminal = -logavgexp(-loss_terminal, dim=-1)
            loss_kl = -logavgexp(-loss_kl_exact, dim=-1)  # Log exact KL loss even when using IWAE, it avoids random negative values
            entropy_prior = dprior.entropy().mean(dim=-1)
            entropy_post = dpost.entropy().mean(dim=-1)

            tensors = dict(loss_kl=loss_kl.detach(),
                           loss_image=loss_image.detach(),
                           entropy_prior=entropy_prior,
                           entropy_post=entropy_post,
                           )

            metrics = dict(loss_wm=loss_model.mean(),
                           loss_wm_image=loss_image.mean(),
                           loss_wm_image_max=loss_image.max(),
                           loss_wm_vecobs=loss_vecobs.mean(),
                           loss_wm_reward=loss_reward.mean(),
                           loss_wm_terminal=loss_terminal.mean(),
                           loss_wm_kl=loss_kl.mean(),
                           loss_wm_kl_max=loss_kl.max(),
                           entropy_prior=entropy_prior.mean(),
                           entropy_post=entropy_post.mean(),
                           )

            tensors['image_rec'] = image_rec.mean(dim=2)  # (N,B,I,C,H,W) => (N,B,C,H,W)

            # Predictions from prior

            if image_pred is not None:
                logprob_img = self.decoder_image.loss(image_pred, image)
                logprob_img = -logavgexp(-logprob_img, dim=-1)  # This is *negative*-log-prob, so actually positive, same as loss
                tensors.update(logprob_img=logprob_img)
                metrics.update(logprob_img=logprob_img.mean())
                tensors['image_pred'] = image_pred.mean(dim=2)

            if reward_pred is not None:
                logprob_reward = self.decoder_reward.loss(reward_pred, reward)
                logprob_reward = -logavgexp(-logprob_reward, dim=-1)
                metrics.update(logprob_reward=logprob_reward.mean())
                tensors.update(reward_pred=reward_pred.mean.mean(dim=-1),  # not quite loss tensor, but fine
                               logprob_reward=logprob_reward)

                mask_pos = (reward.select(-1, 0) > 0)  # mask where reward is *positive*
                logprob_reward_pos = (logprob_reward * mask_pos) / mask_pos  # set to nan where ~mask
                metrics.update(logprob_rewardp=nanmean(logprob_reward_pos))
                tensors.update(logprob_rewardp=logprob_reward_pos)

                mask_neg = (reward.select(-1, 0) < 0)  # mask where reward is *negative*
                logprob_reward_neg = (logprob_reward * mask_neg) / mask_neg  # set to nan where ~mask
                metrics.update(logprob_rewardn=nanmean(logprob_reward_neg))
                tensors.update(logprob_rewardn=logprob_reward_neg)

            if terminal_pred is not None:
                logprob_terminal = self.decoder_terminal.loss(terminal_pred, terminal)
                logprob_terminal = -logavgexp(-logprob_terminal, dim=-1)
                metrics.update(logprob_terminal=logprob_terminal.mean())
                tensors.update(terminal_pred=terminal_pred.mean.mean(dim=-1),    # not quite loss tensor, but fine
                               logprob_terminal=logprob_terminal)

                mask_terminal1 = (terminal.select(-1, 0) == 1)  # mask where terminal is 1
                logprob_terminal1 = (logprob_terminal * mask_terminal1) / mask_terminal1  # set to nan where ~mask
                metrics.update(logprob_terminal1=nanmean(logprob_terminal1))
                tensors.update(logprob_terminal1=logprob_terminal1)

        return loss_model.mean(), features, states, out_state, metrics, tensors
