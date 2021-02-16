import numpy as np
import torch
import torch.nn.functional as F



class MinigridPreprocess:

    def __init__(self, device='cpu', categorical=33):
        self._device = device
        self._categorical = categorical
        self.img_channels = categorical

    def __call__(self, batch):
        # Input:
        #   batch['image']:     np.array(N, B, 7, 7, 1)
        #   batch['image_ids']: np.array(N, B, 7, 7, 3)
        #   batch['action']:    np.array(N, B, 7)
        #   batch['reset']:     np.array(N, B)
        # Output:
        #   image:  torch.tensor(N, B, 33, 7, 7)
        #   action: torch.tensor(N, B, 7)
        #   reset:  torch.tensor(N, B)

        assert batch['image'].shape[-1] == 1

        image = torch.from_numpy(batch['image'][..., 0]).to(torch.int64)
        image = F.one_hot(image, num_classes=self._categorical)
        image = image.permute(0, 1, 4, 2, 3)  # (..., 7, 7, 33) => (..., 33, 7, 7)
        image = image.to(dtype=torch.float, device=self._device)

        action = torch.from_numpy(batch['action']).to(dtype=torch.float, device=self._device)
        reset = torch.from_numpy(batch['reset']).to(dtype=torch.bool, device=self._device)

        return image, action, reset
