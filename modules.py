import torch
import torch.nn as nn
import torch.nn.functional as F


class MinigridEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self._model = nn.Sequential(
            # nn.Conv2d(20, 4, kernel_size=1),  # embedding
            nn.Flatten(-3, -1),
            nn.Linear(20 * 7 * 7, 256),
            nn.ReLU()
        )

    def forward(self, x):
        return self._model(x)


class MinigridDecoder(nn.Module):

    def __init__(self):
        super().__init__()
        self._model = nn.Sequential(
            # nn.Linear(256, 256),
            # nn.ReLU(),
            nn.Linear(256, 20 * 7 * 7),
            nn.Unflatten(-1, (20, 7, 7)),
            # nn.Conv2d(4, 20, kernel_size=1)
        )
        self._loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, x):
        return self._model(x)

    def loss(self, output, target):
        loss = self._loss(output, target).sum(dim=[-1, -2, -3])
        return loss
