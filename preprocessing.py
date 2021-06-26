from typing import Dict, Tuple, Callable
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset


def to_onehot(x: np.ndarray, n_categories) -> np.ndarray:
    e = np.eye(n_categories, dtype=np.float32)
    x = e[x]  # Nice trick: https://stackoverflow.com/a/37323404
    x = x.transpose(0, 1, 4, 2, 3)  # (N, B, H, W, C) => (N, B, C, H, W)
    return x

def to_image(x: np.ndarray) -> np.ndarray:
    assert x.dtype == np.uint8
    x = x.astype(np.float32)
    x = x.transpose(0, 1, 4, 2, 3)  # (N, B, H, W, C) => (N, B, C, H, W)
    x = x / 255.0 - 0.5
    return x


class TransformedDataset(IterableDataset):

    def __init__(self, dataset: IterableDataset, fn: Callable):
        super().__init__()
        self.dataset = dataset
        self.fn = fn

    def __iter__(self):
        for batch in iter(self.dataset):
            yield self.fn(batch)


class MinigridPreprocess:

    def __init__(self, image_categorical=33, image_key='image', map_categorical=33, map_key='map'):
        self._image_categorical = image_categorical
        self._image_key = image_key
        self._map_categorical = map_categorical
        self._map_key = map_key
        self._first = True

    def __call__(self, dataset: IterableDataset) -> IterableDataset:
        return TransformedDataset(dataset, self.apply)

    def apply(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
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
            batch['image'] = to_onehot(batch['image'], self._image_categorical)
        else:
            batch['image'] = to_image(batch['image'])

        if self._map_categorical:
            batch['map'] = to_onehot(batch['map'], self._map_categorical)
        else:
            batch['map'] = to_image(batch['map'])

        assert len(batch['action'].shape) == 3  # should be already one-hot
        batch['action'] = batch['action'].astype(np.float32) 

        if not 'reset' in batch:
            batch['reset'] = np.zeros(batch['action'].shape[0:2], dtype=bool)


        return batch
