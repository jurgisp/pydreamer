import numpy as np
import torch
import torch.nn.functional as F


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
        image = torch.from_numpy(image).to(torch.int64)
        image = F.one_hot(image, num_classes=self._categorical)
        image = image.permute(0, 1, 4, 2, 3)  # (..., 7, 7, 33) => (..., 33, 7, 7)
        image = image.to(dtype=torch.float, device=self._device)

        action = torch.from_numpy(batch['action']).to(dtype=torch.float, device=self._device)

        if 'reset' in batch:
            reset = torch.from_numpy(batch['reset']).to(dtype=torch.bool, device=self._device)
        else:
            reset = torch.zeros(action.shape[0:2], dtype=torch.bool, device=self._device)

        map = torch.from_numpy(batch['map']).to(dtype=torch.int64, device=self._device)

        return image, action, reset, map
