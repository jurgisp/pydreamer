from typing import Any, Tuple

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from models.common import *
from models.functions import *
from models.io import *


class BehavioralCloning(TrainableModel):
    def __init__(self, conf):
        super().__init__()
        self._encoder = ConvEncoder(in_channels=conf.image_channels, cnn_depth=conf.cnn_depth)
        self._actor = MLP(self._encoder.out_dim + 64, conf.action_dim, 400, 4, conf.layer_norm)

    @property
    def submodels(self):
        return (self._encoder, self._actor)

    def optimizers(self, conf):
        optimizer = torch.optim.AdamW(self.parameters(), lr=conf.adam_lr, eps=conf.adam_eps)
        return (optimizer,)

    def grad_clip(self, conf):
        return {
            'grad_norm': nn.utils.clip_grad_norm_(self.parameters(), conf.grad_clip),
        }

    def init_state(self, batch_size: int):
        return None

    def forward(self,
                image: TensorNBCHW,   # (N,B,C,H,W)
                vecobs: Tensor,       # (N,B,V)
                prev_reward: Tensor,  # (N,B)
                prev_action: Tensor,  # (N,B,A)
                reset: Tensor,        # (N,B)
                in_state: Any,
                ):
        e = self._encoder(image)
        y = self._actor.forward(torch.cat((e, vecobs), -1))
        logits = y.log_softmax(-1)
        value = torch.zeros_like(logits).sum(-1)
        out_state = None
        return logits, value, out_state

    def train(self,
              image: TensorNBCHW,
              vecobs: Tensor,
              reward: Tensor,
              terminal: Tensor,
              action_prev: Tensor,
              reset: Tensor,
              map: Tensor,
              map_coord: Tensor,
              map_seen_mask: Tensor,
              in_state: Any,
              I: int = 1,
              H: int = 1,
              imagine_dropout=0,
              do_image_pred=False,
              do_output_tensors=False,
              do_dream_tensors=False,
              ):
        logits, _, _ = self.forward(image, vecobs, reward, action_prev, reset, in_state)
        action_prev = action_prev / (action_prev.sum(-1, keepdim=True) + 1e-6)  # normalize multihot action
        loss = (-(logits[:-1] * action_prev[1:]).sum(-1)).mean()
        entropy = (- (logits * logits.exp()).sum(-1)).mean()
        metrics = {'loss': loss.detach(), 'policy_entropy': entropy.detach()}
        return (loss,), metrics, {}, None, {}, {}
