import numpy as np
import torch
import torch.nn.functional as F


def to_onehot(x_np, n_categories):
    x = torch.from_numpy(x_np).to(torch.int64)
    x = F.one_hot(x, num_classes=n_categories)
    x = x.permute(0, 1, 4, 2, 3)  # (N, B, H, W, C) => (N, B, C, H, W)
    x = x.to(dtype=torch.float)
    return x


class MinigridPreprocess:

    def __init__(self, device='cpu', image_categorical=33, image_key='image', map_categorical=33, map_key='map'):
        self._device = device
        self._image_categorical = image_categorical
        self._image_key = image_key
        self._map_categorical = map_categorical
        self._map_key = map_key
        self._first = True

    def __call__(self, batch):
        # Input:
        #   batch['image']:     np.array(N, B, 7, 7)
        #   batch['action']:    np.array(N, B, 7)
        #   batch['reset']:     np.array(N, B)
        # Output:
        #   image:  torch.tensor(N, B, 33, 7, 7)
        #   action: torch.tensor(N, B, 7)
        #   reset:  torch.tensor(N, B)
        #   map:    torch.tensor(N, B, 7, 7)

        if self._first:
            print('Data batch: ', {k: v.shape for k, v in batch.items()})
            self._first = False

        batch['image'] = batch[self._image_key]  # Use something else (e.g. map_masked) as image
        batch['map'] = batch[self._map_key]

        if self._image_categorical:
            image = to_onehot(batch['image'], self._image_categorical).to(device=self._device)
        else:
            image = torch.from_numpy(batch['image'])
            image = image.permute(0, 1, 4, 2, 3)  # (N, B, H, W, C) => (N, B, C, H, W)
            image= image.to(dtype=torch.float, device=self._device)

        if self._map_categorical:
            map = to_onehot(batch['map'], self._map_categorical).to(device=self._device)
        else:
            map = torch.from_numpy(batch['map'])
            map = map.permute(0, 1, 4, 2, 3)  # (N, B, H, W, C) => (N, B, C, H, W)
            map= map.to(dtype=torch.float, device=self._device)

        action = torch.from_numpy(batch['action']).to(dtype=torch.float, device=self._device)

        if 'reset' in batch:
            reset = torch.from_numpy(batch['reset']).to(dtype=torch.bool, device=self._device)
        else:
            reset = torch.zeros(action.shape[0:2], dtype=torch.bool, device=self._device)

        return image, action, reset, map
