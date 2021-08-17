import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch import Tensor

from modules_tools import *
from modules_common import *


class ActorCritic(nn.Module):

    def __init__(self, in_dim, out_actions, hidden_dim=400, hidden_layers=4, discount=0.999, discount_lambda=0.95, temperature=1e-3, target_interval=100):
        super().__init__()
        self.in_dim = in_dim
        self.out_actions = out_actions
        self._gamma = discount
        self._lambda = discount_lambda
        self._temperature = temperature
        self._target_interval = target_interval
        self._actor = MLP(in_dim, out_actions, hidden_dim, hidden_layers)
        self._critic = MLP(in_dim, 1, hidden_dim, hidden_layers)
        self._critic_target = MLP(in_dim, 1, hidden_dim, hidden_layers)
        self._train_steps = 0

    def forward_act(self, features: Tensor) -> D.OneHotCategorical:
        y = self._actor.forward(features)
        p = D.OneHotCategorical(logits=y)
        return p

    def train(self, features: TensorJMF, rewards: D.Distribution, terminals: D.Distribution, actions: TensorHMA):
        if self._train_steps % self._target_interval == 0:
            self.update_critic_target()
        self._train_steps += 1

        reward1: TensorHM = rewards.mean[1:]
        terminal0: TensorHM = terminals.mean[:-1]
        terminal1: TensorHM = terminals.mean[1:]
        policy = self.forward_act(features[:-1])
        value0: TensorHM = self._critic.forward(features[:-1])
        value0t: TensorHM = self._critic_target.forward(features[:-1]).detach()
        value1t: TensorHM = self._critic_target.forward(features[1:]).detach()
        advantage = - value0t + reward1 + self._gamma * (1.0 - terminal1) * value1t
        assert not advantage.requires_grad

        # GAE from https://arxiv.org/abs/1506.02438 eq (16)
        #   advantage_gae[t] = advantage[t] + (gamma lambda) advantage[t+1] + (gamma lambda)^2 advantage[t+2] + ...
        advantage_gae = []
        agae = None
        for adv, term in zip(reversed(advantage.unbind()), reversed(terminal1.unbind())):
            if agae is None:
                agae = adv
            else:
                agae = adv + self._lambda * self._gamma * (1.0 - term) * agae
            advantage_gae.append(agae)
        advantage_gae.reverse()
        advantage_gae = torch.stack(advantage_gae)
        assert not advantage_gae.requires_grad

        # Sanity check #1: if lambda=0, then advantage_gae=advantage, then
        #   value_target = advantage + value0t
        #                = reward + gamma * value1t
        value_target = advantage_gae + value0t

        # Sanity check #2: if lambda=1 then
        #   advantage_gae[t] = value_target_mc[t]
        #                    = reward[t] + g[t] reward[t+1] + g[t]g[t+1] reward[t+2] + ... + g[t]g[t+1]g[t+2]g[..] value1t[-1]
        # where
        #   g[t] = gamma * (1 - terminal[t])
        # is the adjusted discount

        # if self._lambda == 1:
        #     value_target_mc = []
        #     v = value1t[-1]
        #     for rew, term in zip(reversed(reward.unbind()), reversed(terminal1.unbind())):
        #         v = rew + self._gamma * (1.0 - term) * v
        #         value_target_mc.append(v)
        #     value_target_mc.reverse()
        #     value_target_mc = torch.stack(value_target_mc)
        #     assert torch.allclose(value_target, value_target_mc)

        # When calculating losses, should ignore terminal states, or anything after, so:
        #   reality_weight[i] = (1-terminal[0]) (1-terminal[1]) ... (1-terminal[i])
        # Note this takes care of the case when initial state features[0] is terminal - it will get weighted by (1-terminals[0]).
        reality_weight = (1 - terminal0).log().cumsum(dim=0).exp()

        loss_value = torch.square(value_target - value0)
        loss_value = (loss_value * reality_weight).mean()

        loss_policy = - (policy.logits * actions).sum(-1) * advantage
        loss_policy = (loss_policy * reality_weight).mean()

        policy_entropy = (policy.entropy() * reality_weight).mean()
        loss_entropy = - self._temperature * policy_entropy

        with torch.no_grad():
            metrics = dict(loss_ac_value=loss_value.detach(),
                           loss_ac_policy=loss_policy.detach(),
                           policy_entropy=policy_entropy.detach(),
                           policy_value=value0[0].mean(),  # Value of real states
                           policy_value_im=value0.mean(),  # Value of imagined states
                           policy_reward=reward1.mean(),
                           policy_reward_std=reward1.std(),
                           )
            tensors = dict(policy_value=value0[0].detach())

        loss = loss_value + loss_policy + loss_entropy
        return loss, metrics, tensors

    def update_critic_target(self):
        self._critic_target.load_state_dict(self._critic.state_dict())  # type: ignore
