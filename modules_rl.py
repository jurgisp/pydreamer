import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch import Tensor

from modules_tools import *
from modules_common import *


class ActorCritic(nn.Module):

    def __init__(self, in_dim, out_actions, hidden_dim=400, hidden_layers=4, gamma=0.99, temperature=1e-3, target_update_interval=100):
        super().__init__()
        self.in_dim = in_dim
        self.out_actions = out_actions
        self._gamma = gamma
        self._temperature = temperature
        self._target_update_interval = target_update_interval
        self._actor = MLP(in_dim, out_actions, hidden_dim, hidden_layers)
        self._critic = MLP(in_dim, 1, hidden_dim, hidden_layers)
        self._critic_target = MLP(in_dim, 1, hidden_dim, hidden_layers)
        self._train_steps = 0

    def forward_act(self, features: Tensor) -> D.Categorical:
        y = self._actor(features)
        p = D.Categorical(logits=y)
        return p

    def forward_act_sample(self, features: Tensor) -> Tensor:
        p = self.forward_act(features)
        a = F.one_hot(p.sample(), num_classes=self.out_actions)
        a = a.to(torch.float)
        return a

    def train(self, features: TensorJMF, rewards: TensorJM2, terminals: TensorJM2, actions: TensorHMA) -> Tuple[Tensor, Dict[str, Tensor]]:
        if self._train_steps % self._target_update_interval == 0:
            self.update_critic_target()
        self._train_steps += 1

        reward: TensorHM = F.softmax(rewards[1:], -1).select(-1, 1)  # select(-1,1) => probability of getting reward=1
        terminal: TensorHM = F.softmax(terminals[1:], -1).select(-1, 1)  # select(-1,) => probability of terminal state
        policy = self.forward_act(features[:-1])
        value: TensorHM = self._critic.forward(features[:-1])
        value_baseline: TensorHM = self._critic_target.forward(features[:-1]).detach()
        value_target: TensorHM = self._critic_target.forward(features[1:]).detach()

        target: TensorHM = reward + self._gamma * value_target * (1.0 - terminal)
        assert not target.requires_grad
        loss_value = torch.square(value - target)
        loss_value = loss_value.mean()

        advantage = target - value_baseline
        assert not advantage.requires_grad
        loss_critic = - (policy.logits * actions).sum(-1) * advantage
        loss_critic = loss_critic.mean()

        policy_entropy = policy.entropy().mean()
        loss_entropy = - self._temperature * policy_entropy

        # TODO: need weights, even for H=1, weight=terminals[0]
        # TODO: lambda-GAE for H>1

        with torch.no_grad():
            metrics = dict(loss_ac_value=loss_value.detach(),
                           loss_ac_critic=loss_critic.detach(),
                           policy_entropy=policy_entropy.detach(),
                           policy_value=value.mean()
                           )

        loss = loss_value + loss_critic + loss_entropy
        return loss, metrics

    def update_critic_target(self):
        self._critic_target.load_state_dict(self._critic.state_dict())  # type: ignore