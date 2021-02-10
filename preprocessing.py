import numpy as np
import torch


MINIGRID_VALUES = np.array([
    # Invisible
    [0, 0, 0],
    # Empty
    [1, 0, 0],
    # Wall
    [2, 5, 0],
    # Door (color, open)
    [4, 0, 0],
    [4, 0, 1],
    [4, 1, 0],
    [4, 1, 1],
    [4, 2, 0],
    [4, 2, 1],
    [4, 3, 0],
    [4, 3, 1],
    [4, 4, 0],
    [4, 4, 1],
    [4, 5, 0],
    [4, 5, 1],
    # Key (color)
    [5, 0, 0],
    [5, 1, 0],
    [5, 2, 0],
    [5, 3, 0],
    [5, 4, 0],
    [5, 5, 0],
    # Ball (color)
    [6, 0, 0],
    [6, 1, 0],
    [6, 2, 0],
    [6, 3, 0],
    [6, 4, 0],
    [6, 5, 0],
    # Box (color)
    [7, 0, 0],
    [7, 1, 0],
    [7, 2, 0],
    [7, 3, 0],
    [7, 4, 0],
    [7, 5, 0]])


class MinigridPreprocess:

    def __init__(self, device='cpu', categorical=False):
        self._device = device
        self._categorical = categorical
        self.img_channels = len(MINIGRID_VALUES) if categorical else 20

    def __call__(self, batch):
        # Input:
        #   batch['image']:     np.array(N, B, 7, 7, 20)
        #   batch['image_ids']: np.array(N, B, 7, 7, 3)
        #   batch['action']:    np.array(N, B, 7)
        #   batch['reset']:     np.array(N, B)
        # Output:
        #   image:  torch.tensor(N, B, 20|33, 7, 7)
        #   action: torch.tensor(N, B, 7)
        #   reset:  torch.tensor(N, B)

        if self._categorical:
            image = self.grid_to_categorical(batch['image_ids']).transpose(0, 1, 4, 2, 3)
        else:
            image = batch['image'].transpose(0, 1, 4, 2, 3) > 0
        image = torch.from_numpy(image).to(dtype=torch.float, device=self._device)

        action = torch.from_numpy(batch['action']).to(dtype=torch.float, device=self._device)
        reset = torch.from_numpy(batch['reset']).to(dtype=torch.bool, device=self._device)

        return image, action, reset

    def grid_to_categorical(self, image_ids):
        n = len(MINIGRID_VALUES)
        out = np.zeros(image_ids.shape[:-1] + (n,))
        for i in range(n):
            val = MINIGRID_VALUES[i]
            out[..., i] = (image_ids == val).all(axis=-1)
        return out
