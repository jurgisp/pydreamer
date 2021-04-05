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

    def __init__(self, device='cpu', categorical=33, image_key='image', map_key='map'):
        self._device = device
        self._categorical = categorical
        self.img_channels = categorical
        self._image_key = image_key
        self._map_key = map_key

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

        batch['image'] = batch[self._image_key]  # Use something else (e.g. map_masked) as image
        batch['map'] = batch[self._map_key]

        image = batch['image']
        assert image.shape[-1] == 7, f'Unexpected image shape {image.shape}'
        image = to_onehot(image, self._categorical).to(device=self._device)

        action = torch.from_numpy(batch['action']).to(dtype=torch.float, device=self._device)

        if 'reset' in batch:
            reset = torch.from_numpy(batch['reset']).to(dtype=torch.bool, device=self._device)
        else:
            reset = torch.zeros(action.shape[0:2], dtype=torch.bool, device=self._device)

        map = to_onehot(batch['map'], self._categorical).to(device=self._device)

        return image, action, reset, map
