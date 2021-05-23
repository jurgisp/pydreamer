from typing import Tuple, Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class GRU2Inputs(nn.Module):

    def __init__(self, input1_dim, input2_dim, mlp_dim=200, state_dim=200, num_layers=1, bidirectional=False, input_activation=F.elu):
        super().__init__()
        self._in_mlp1 = nn.Linear(input1_dim, mlp_dim)
        self._in_mlp2 = nn.Linear(input2_dim, mlp_dim, bias=False)
        self._act = input_activation
        self._gru = nn.GRU(input_size=mlp_dim, hidden_size=state_dim, num_layers=num_layers, bidirectional=bidirectional)
        self._directions = 2 if bidirectional else 1

    def init_state(self, batch_size):
        device = next(self._gru.parameters()).device
        return torch.zeros((
            self._gru.num_layers * self._directions, 
            batch_size, 
            self._gru.hidden_size), device=device)

    def forward(self,
                input1_seq: Tensor,  # (N,B,X1)
                input2_seq: Tensor,  # (N,B,X2)
                in_state: Optional[Tensor] = None
                ) -> Tuple[Tensor, Tensor]:
        if in_state is None:
            in_state = self.init_state(input1_seq.size(1))
        inp = self._act(self._in_mlp1(input1_seq) + self._in_mlp2(input2_seq))
        output, out_state = self._gru(inp, in_state)
        # NOTE: Different from nn.GRU: detach output state
        return output, out_state.detach()
