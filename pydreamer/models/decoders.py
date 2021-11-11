from typing import Optional, Union
import torch
import torch.nn as nn
import torch.distributions as D

from models.functions import *
from models.common import *


class ConvDecoder(nn.Module):

    def __init__(self,
                 in_dim,
                 out_channels=3,
                 cnn_depth=32,
                 mlp_layers=0,
                 layer_norm=True,
                 activation=nn.ELU
                 ):
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

    def forward(self, x: Tensor) -> Tensor:
        x, bd = flatten_batch(x)
        y = self.model(x)
        y = unflatten_batch(y, bd)
        return y

    def loss(self, output: Tensor, target: Tensor) -> Tensor:
        output, bd = flatten_batch(output, 3)
        target, _ = flatten_batch(target, 3)
        loss = 0.5 * torch.square(output - target).sum(dim=[-1, -2, -3])  # MSE
        return unflatten_batch(loss, bd)

    def training_step(self, features: TensorNBIF, target: TensorNBCHW) -> Tuple[TensorNBI, TensorNB, TensorNBCHW]:
        assert len(features.shape) == 4 and len(target.shape) == 5
        I = features.shape[2]
        target = insert_dim(target, 2, I)  # Expand target with iwae_samples dim, because features have it

        decoded = self.forward(features)
        loss_nbi = self.loss(decoded, target)
        loss_nb = -logavgexp(-loss_nbi, dim=2)  # NBI => NB
        decoded = decoded.mean(dim=2)  # NBICHW => NBCHW

        assert len(loss_nbi.shape) == 3 and len(decoded.shape) == 5
        return loss_nbi, loss_nb, decoded


class CatImageDecoder(nn.Module):
    """Dense decoder for categorical image, e.g. map"""

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

    def loss(self, output: Tensor, target: Tensor) -> Tensor:
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

    def training_step(self, features: TensorNBIF, target: TensorNBCHW) -> Tuple[TensorNBI, TensorNB, TensorNBCHW]:
        assert len(features.shape) == 4 and len(target.shape) == 5
        I = features.shape[2]
        target = insert_dim(target, 2, I)  # Expand target with iwae_samples dim, because features have it

        logits = self.forward(features)
        loss_nbi = self.loss(logits, target)
        loss_nb = -logavgexp(-loss_nbi, dim=2)  # NBI => NB

        assert len(logits.shape) == 6   # NBICHW
        logits = logits - logits.logsumexp(dim=-3, keepdim=True)  # normalize C
        logits = torch.logsumexp(logits, dim=2)  # aggregate I => NBCHW
        logits = logits - logits.logsumexp(dim=-3, keepdim=True)  # normalize C again
        decoded = logits

        assert len(loss_nbi.shape) == 3 and len(decoded.shape) == 5
        return loss_nbi, loss_nb, decoded


class DenseBernoulliDecoder(nn.Module):

    def __init__(self, in_dim, hidden_dim=400, hidden_layers=2, layer_norm=True):
        super().__init__()
        self.model = MLP(in_dim, 1, hidden_dim, hidden_layers, layer_norm)

    def forward(self, features: Tensor) -> D.Distribution:
        y = self.model.forward(features)
        p = D.Bernoulli(logits=y.float())
        return p

    def loss(self, output: D.Distribution, target: Tensor) -> Tensor:
        return -output.log_prob(target)

    def training_step(self, features: TensorNBIF, target: Tensor) -> Tuple[TensorNBI, TensorNB, TensorNB]:
        assert len(features.shape) == 4
        I = features.shape[2]
        target = insert_dim(target, 2, I)  # Expand target with iwae_samples dim, because features have it

        decoded = self.forward(features)
        loss_nbi = self.loss(decoded, target)
        loss_nb = -logavgexp(-loss_nbi, dim=2)  # NBI => NB
        decoded = decoded.mean.mean(dim=2)

        assert len(loss_nbi.shape) == 3
        assert len(loss_nb.shape) == 2
        assert len(decoded.shape) == 2
        return loss_nbi, loss_nb, decoded


class DenseNormalDecoder(nn.Module):

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

    def training_step(self, features: TensorNBIF, target: Tensor) -> Tuple[TensorNBI, TensorNB, Tensor]:
        assert len(features.shape) == 4
        I = features.shape[2]
        target = insert_dim(target, 2, I)  # Expand target with iwae_samples dim, because features have it

        decoded = self.forward(features)
        loss_nbi = self.loss(decoded, target)
        loss_nb = -logavgexp(-loss_nbi, dim=2)  # NBI => NB
        decoded = decoded.mean.mean(dim=2)

        assert len(loss_nbi.shape) == 3
        assert len(loss_nb.shape) == 2
        assert len(decoded.shape) == (2 if self.out_dim == 1 else 3)
        return loss_nbi, loss_nb, decoded


class DenseCategoricalSupportDecoder(nn.Module):
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

    def training_step(self, features: TensorNBIF, target: Tensor) -> Tuple[TensorNBI, TensorNB, TensorNB]:
        assert len(features.shape) == 4
        I = features.shape[2]
        target = insert_dim(target, 2, I)  # Expand target with iwae_samples dim, because features have it

        decoded = self.forward(features)
        loss_nbi = self.loss(decoded, target)
        loss_nb = -logavgexp(-loss_nbi, dim=2)  # NBI => NB
        decoded = decoded.mean.mean(dim=2)

        assert len(loss_nbi.shape) == 3
        assert len(loss_nb.shape) == 2
        assert len(decoded.shape) == 2
        return loss_nbi, loss_nb, decoded
