from typing import Optional, Union
import torch
import torch.nn as nn
import torch.distributions as D

from models.functions import *
from models.common import *


class ConvEncoder(nn.Module):

    def __init__(self, in_channels=3, cnn_depth=32, activation=nn.ELU):
        super().__init__()
        self.out_dim = cnn_depth * 32
        kernels = (4, 4, 4, 4)
        stride = 2
        d = cnn_depth
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, d, kernels[0], stride),
            activation(),
            nn.Conv2d(d, d * 2, kernels[1], stride),
            activation(),
            nn.Conv2d(d * 2, d * 4, kernels[2], stride),
            activation(),
            nn.Conv2d(d * 4, d * 8, kernels[3], stride),
            activation(),
            nn.Flatten()
        )

    def forward(self, x):
        x, bd = flatten_batch(x, 3)
        y = self.model(x)
        y = unflatten_batch(y, bd)
        return y


class ConvDecoder(nn.Module):

    def __init__(self, in_dim, out_channels=3, cnn_depth=32, mlp_layers=0, layer_norm=True, activation=nn.ELU):
        super().__init__()
        self.in_dim = in_dim
        kernels = (5, 5, 6, 6)
        stride = 2
        d = cnn_depth
        if mlp_layers == 0:
            layers = [
                nn.Linear(in_dim, d * 32),  # No activation here in DreamerV2
            ]
        else:
            hidden_dim = d * 32
            norm = nn.LayerNorm if layer_norm else NoNorm
            layers = [
                nn.Linear(in_dim, hidden_dim),
                norm(hidden_dim, eps=1e-3),
                activation()
            ]
            for _ in range(mlp_layers - 1):
                layers += [
                    nn.Linear(hidden_dim, hidden_dim),
                    norm(hidden_dim, eps=1e-3),
                    activation()]

        self.model = nn.Sequential(
            # FC
            *layers,
            nn.Unflatten(-1, (d * 32, 1, 1)),  # type: ignore
            # Deconv
            nn.ConvTranspose2d(d * 32, d * 4, kernels[0], stride),
            activation(),
            nn.ConvTranspose2d(d * 4, d * 2, kernels[1], stride),
            activation(),
            nn.ConvTranspose2d(d * 2, d, kernels[2], stride),
            activation(),
            nn.ConvTranspose2d(d, out_channels, kernels[3], stride))

    def forward(self, x):
        x, bd = flatten_batch(x)
        y = self.model(x)
        y = unflatten_batch(y, bd)
        return y

    def loss(self, output, target):
        output, bd = flatten_batch(output, 3)
        target, _ = flatten_batch(target, 3)
        loss = 0.5 * torch.square(output - target).sum(dim=[-1, -2, -3])  # MSE
        return unflatten_batch(loss, bd)

    def to_distr(self, output: Tensor) -> Tensor:  # Return mean
        assert len(output.shape) == 6  # (N,B,I,C,H,W)
        return output.mean(dim=2)  # (N,B,I,C,H,W) => (N,B,C,H,W)


class DenseEncoder(nn.Module):

    def __init__(self, in_dim, out_dim=256, activation=nn.ELU, hidden_dim=400, hidden_layers=2, layer_norm=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        norm = nn.LayerNorm if layer_norm else NoNorm
        layers = [nn.Flatten()]
        layers += [
            nn.Linear(in_dim, hidden_dim),
            norm(hidden_dim, eps=1e-3),
            activation()]
        for _ in range(hidden_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                norm(hidden_dim, eps=1e-3),
                activation()]
        layers += [
            nn.Linear(hidden_dim, out_dim),
            activation()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x, bd = flatten_batch(x, 3)
        y = self.model(x)
        y = unflatten_batch(y, bd)
        return y


class DenseBernoulliHead(nn.Module):

    def __init__(self, in_dim, hidden_dim=400, hidden_layers=2, layer_norm=True):
        super().__init__()
        self.model = MLP(in_dim, 1, hidden_dim, hidden_layers, layer_norm)

    def forward(self, features: Tensor) -> D.Distribution:
        y = self.model.forward(features)
        p = D.Bernoulli(logits=y.float())
        return p

    def loss(self, output: D.Distribution, target: Tensor) -> Tensor:
        return -output.log_prob(target)


class DenseNormalHead(nn.Module):

    def __init__(self, in_dim, out_dim=1, hidden_dim=400, hidden_layers=2, layer_norm=True, std=0.3989422804):
        super().__init__()
        self.model = MLP(in_dim, out_dim, hidden_dim, hidden_layers, layer_norm)
        self.std = std
        self.out_dim = out_dim

    def forward(self, features: Tensor) -> D.Distribution:
        y = self.model.forward(features)
        p = D.Normal(loc=y, scale=torch.ones_like(y) * self.std)
        if self.out_dim > 1:
            p = D.independent.Independent(p, 1)  # Makes p.logprob() sum over last dim
        return p

    def loss(self, output: D.Distribution, target: Tensor) -> Tensor:
        var = self.std ** 2  # var cancels denominator, which makes loss = 0.5 (target-output)^2
        return -output.log_prob(target) * var


class DenseCategoricalSupportHead(nn.Module):
    """
    Represent continuous variable distribution by discrete set of support values.
    Useful for reward head, which can be e.g. [-10, 0, 1, 10]
    """

    def __init__(self, in_dim, support=[0.0, 1.0], hidden_dim=400, hidden_layers=2, layer_norm=True):
        assert isinstance(support, list)
        super().__init__()
        self.model = MLP(in_dim, len(support), hidden_dim, hidden_layers, layer_norm)
        self.support = nn.Parameter(torch.tensor(support), requires_grad=False)

    def forward(self, features: Tensor) -> D.Distribution:
        y = self.model.forward(features)
        p = CategoricalSupport(logits=y.float(), support=self.support.data)
        return p

    def loss(self, output: D.Distribution, target: Tensor) -> Tensor:
        target = self.to_categorical(target)
        return -output.log_prob(target)

    def to_categorical(self, target: Tensor) -> Tensor:
        # TODO: should interpolate between adjacent values, like in MuZero
        distances = torch.square(target.unsqueeze(-1) - self.support)
        return distances.argmin(-1)


class DenseDecoder(nn.Module):

    def __init__(self, in_dim, out_shape=(33, 7, 7), activation=nn.ELU, hidden_dim=400, hidden_layers=2, layer_norm=True, min_prob=0):
        super().__init__()
        self.in_dim = in_dim
        self.out_shape = out_shape
        norm = nn.LayerNorm if layer_norm else NoNorm
        layers = []
        layers += [
            nn.Linear(in_dim, hidden_dim),
            norm(hidden_dim, eps=1e-3),
            activation()]
        for _ in range(hidden_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                norm(hidden_dim, eps=1e-3),
                activation()]
        layers += [
            nn.Linear(hidden_dim, np.prod(out_shape)),
            nn.Unflatten(-1, out_shape)]
        self.model = nn.Sequential(*layers)
        self.min_prob = min_prob

    def forward(self, x: Tensor) -> Tensor:
        x, bd = flatten_batch(x)
        y = self.model(x)
        y = unflatten_batch(y, bd)
        return y

    def loss(self,
             output,  # (*,C,H,W)
             target   # float(*,C,H,W) or int(*,H,W)
             ):
        if len(output.shape) == len(target.shape):
            target = target.argmax(dim=-3)  # float(*,C,H,W) => int(*,H,W)
        assert target.dtype == torch.int64, 'Target should be categorical'
        output, bd = flatten_batch(output, len(self.out_shape))     # (*,C,H,W) => (B,C,H,W)
        target, _ = flatten_batch(target, len(self.out_shape) - 1)  # (*,H,W) => (B,H,W)

        if self.min_prob == 0:
            loss = F.nll_loss(F.log_softmax(output, 1), target, reduction='none')  # = F.cross_entropy()
        else:
            prob = F.softmax(output, 1)
            prob = (1.0 - self.min_prob) * prob + self.min_prob * (1.0 / prob.size(1))  # mix with uniform prob
            loss = F.nll_loss(prob.log(), target, reduction='none')

        if len(self.out_shape) == 3:
            loss = loss.sum(dim=[-1, -2])  # (*,H,W) => (*)
        assert len(loss.shape) == 1
        return unflatten_batch(loss, bd)

    def accuracy(self, output: TensorNBICHW, target: Union[TensorNBICHW, IntTensorNBIHW], map_seen_mask: Optional[Tensor] = None):
        if len(output.shape) == len(target.shape):
            target = target.argmax(dim=-3)  # float(*,I,C,H,W) => int(*,I,H,W)
        output, bd = flatten_batch(output, 4)
        target, _ = flatten_batch(target, 3)

        output = -logavgexp(-output, dim=-4)  # (*,I,C,H,W) => (*,C,H,W)
        target = target.select(-3, 0)  # int(*,I,H,W) => int(*,H,W)

        acc = output.argmax(dim=-3) == target
        if map_seen_mask is None:
            acc = acc.to(torch.float).mean([-1, -2])
        else:
            map_seen_mask, _ = flatten_batch(map_seen_mask, 2)  # (*,H,W)
            acc = (acc * map_seen_mask).sum([-1, -2]) / map_seen_mask.sum([-1, -2])
        acc = unflatten_batch(acc, bd)  # (N,B)
        return acc

    def to_distr(self, output: Tensor) -> Tensor:  # Return logits
        assert len(output.shape) == 6
        logits = output.permute(2, 0, 1, 4, 5, 3)  # (N,B,I,C,H,W) => (I,N,B,H,W,C)
        # Normalize probability
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        # Aggregate prob=avg(prob_i)
        logits_agg = torch.logsumexp(logits, dim=0)  # (I,N,B,H,W,C) => (N,B,H,W,C)
        logits_agg = D.Categorical(logits=logits_agg).logits  # Normalize
        return logits_agg.permute(0, 1, 4, 2, 3)  # (N,B,H,W,C) => (N,B,C,H,W)
