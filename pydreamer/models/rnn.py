from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from torch import Tensor
from torch.nn import Parameter


class GRU2Inputs(nn.Module):

    def __init__(self, input1_dim, input2_dim, mlp_dim=200, state_dim=200, num_layers=1, bidirectional=False, input_activation=F.elu):
        super().__init__()
        self.in_mlp1 = nn.Linear(input1_dim, mlp_dim)
        self.in_mlp2 = nn.Linear(input2_dim, mlp_dim, bias=False)
        self.act = input_activation
        self.gru = nn.GRU(input_size=mlp_dim, hidden_size=state_dim, num_layers=num_layers, bidirectional=bidirectional)
        self.directions = 2 if bidirectional else 1

    def init_state(self, batch_size):
        device = next(self.gru.parameters()).device
        return torch.zeros((
            self.gru.num_layers * self.directions,
            batch_size,
            self.gru.hidden_size), device=device)

    def forward(self,
                input1_seq: Tensor,  # (T,B,X1)
                input2_seq: Tensor,  # (T,B,X2)
                in_state: Optional[Tensor] = None
                ) -> Tuple[Tensor, Tensor]:
        if in_state is None:
            in_state = self.init_state(input1_seq.size(1))
        inp = self.act(self.in_mlp1(input1_seq) + self.in_mlp2(input2_seq))
        output, out_state = self.gru(inp, in_state)
        # NOTE: Different from nn.GRU: detach output state
        return output, out_state.detach()


class GRUCellStack(nn.Module):
    """Multi-layer stack of GRU cells"""

    def __init__(self, input_size, hidden_size, num_layers, cell_type):
        super().__init__()
        self.num_layers = num_layers
        layer_size = hidden_size // num_layers
        assert layer_size * num_layers == hidden_size, "Must be divisible"
        if cell_type == 'gru':
            cell = nn.GRUCell
        elif cell_type == 'gru_layernorm':
            cell = NormGRUCell
        elif cell_type == 'gru_layernorm_dv2':
            cell = NormGRUCellLateReset
        else:
            assert False, f'Unknown cell type {cell_type}'
        layers = [cell(input_size, layer_size)] 
        layers.extend([cell(layer_size, layer_size) for _ in range(num_layers - 1)])
        self.layers = nn.ModuleList(layers)

    def forward(self, input: Tensor, state: Tensor) -> Tensor:
        input_states = state.chunk(self.num_layers, -1)
        output_states = []
        x = input
        for i in range(self.num_layers):
            x = self.layers[i](x, input_states[i])
            output_states.append(x)
        return torch.cat(output_states, -1)


class GRUCell(jit.ScriptModule):
    """Reproduced regular nn.GRUCell, for reference"""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(input_size, 3 * hidden_size))
        self.weight_hh = Parameter(torch.randn(hidden_size, 3 * hidden_size))
        self.bias_ih = Parameter(torch.randn(3 * hidden_size))
        self.bias_hh = Parameter(torch.randn(3 * hidden_size))

    @jit.script_method
    def forward(self, input: Tensor, state: Tensor) -> Tensor:
        gates_i = torch.mm(input, self.weight_ih) + self.bias_ih
        gates_h = torch.mm(state, self.weight_hh) + self.bias_hh
        reset_i, update_i, newval_i = gates_i.chunk(3, 1)
        reset_h, update_h, newval_h = gates_h.chunk(3, 1)
        reset = torch.sigmoid(reset_i + reset_h)
        update = torch.sigmoid(update_i + update_h)
        newval = torch.tanh(newval_i + reset * newval_h)
        h = update * newval + (1 - update) * state
        return h


class NormGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Linear(input_size, 3 * hidden_size, bias=False)
        self.weight_hh = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.ln_reset = nn.LayerNorm(hidden_size, eps=1e-3)
        self.ln_update = nn.LayerNorm(hidden_size, eps=1e-3)
        self.ln_newval = nn.LayerNorm(hidden_size, eps=1e-3)

    def forward(self, input: Tensor, state: Tensor) -> Tensor:
        gates_i = self.weight_ih(input)
        gates_h = self.weight_hh(state)
        reset_i, update_i, newval_i = gates_i.chunk(3, 1)
        reset_h, update_h, newval_h = gates_h.chunk(3, 1)

        reset = torch.sigmoid(self.ln_reset(reset_i + reset_h))
        update = torch.sigmoid(self.ln_update(update_i + update_h))
        newval = torch.tanh(self.ln_newval(newval_i + reset * newval_h))
        h = update * newval + (1 - update) * state
        return h


class NormGRUCellLateReset(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Linear(input_size, 3 * hidden_size, bias=False)
        self.weight_hh = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.lnorm = nn.LayerNorm(3 * hidden_size, eps=1e-3)
        self.update_bias = -1

    def forward(self, input: Tensor, state: Tensor) -> Tensor:
        gates = self.weight_ih(input) + self.weight_hh(state)
        gates = self.lnorm(gates)
        reset, update, newval = gates.chunk(3, 1)

        reset = torch.sigmoid(reset)
        update = torch.sigmoid(update + self.update_bias)
        newval = torch.tanh(reset * newval)  # late reset, diff from normal GRU
        h = update * newval + (1 - update) * state
        return h


class LSTMCell(jit.ScriptModule):
    # Example from https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)
