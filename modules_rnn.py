import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU2Inputs(nn.Module):

    def __init__(self, input1_dim, input2_dim, input_dim=200, state_dim=200, num_layers=1, input_activation=F.elu):
        super().__init__()
        self._in_mlp1 = nn.Linear(input1_dim, input_dim)
        self._in_mlp2 = nn.Linear(input2_dim, input_dim, bias=False)
        self._act = input_activation
        self._gru = nn.GRU(input_size=input_dim, hidden_size=state_dim, num_layers=num_layers)

    def forward(self, input1_seq, input2_seq, in_state):
        inp = self._act(self._in_mlp1(input1_seq) + self._in_mlp2(input2_seq))
        output, out_state = self._gru(inp, in_state)
        return output, out_state
