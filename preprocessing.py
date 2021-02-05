import torch


class MinigridPreprocess:

    def __init__(self, device='cpu'):
        self._device = device

    def __call__(self, batch):
        # Input:
        #   batch['image']: np.array(N, B, 7, 7, 20)
        # Output:
        #   image: torch.tensor(N, B, 20, 7, 7)

        image = batch['image'].transpose(0, 1, 4, 2, 3) > 0
        image = torch.from_numpy(image).to(dtype=torch.float, device=self._device)

        return image