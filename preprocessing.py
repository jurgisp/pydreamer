from typing import Dict, Tuple, Callable
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info


def to_onehot(x: np.ndarray, n_categories) -> np.ndarray:
    e = np.eye(n_categories, dtype=np.float32)
    x = e[x]  # Nice trick: https://stackoverflow.com/a/37323404
    x = x.transpose(0, 1, 4, 2, 3)  # (N, B, H, W, C) => (N, B, C, H, W)
    return x

def to_image(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        x = x.astype(np.float32)
        x = x / 255.0 - 0.5
    else:
        assert 0.0 <= x[0,0,0,0,0] and x[0,0,0,0,0] <= 1.0
        x = x.astype(np.float32)
    x = x.transpose(0, 1, 4, 2, 3)  # (N, B, H, W, C) => (N, B, C, H, W)
    return x


class WorkerInfoPreprocess(IterableDataset):

    def __init__(self, dataset: IterableDataset):
        super().__init__()
        self.dataset = dataset

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info:
            worker_id = worker_info.id
            print(f'Started data worker ({worker_id})')
        else:
            worker_id = 0
        for batch in iter(self.dataset):
            yield batch, worker_id


class TransformedDataset(IterableDataset):

    def __init__(self, dataset: IterableDataset, fn: Callable):
        super().__init__()
        self.dataset = dataset
        self.fn = fn

    def __iter__(self):
        for batch in iter(self.dataset):
            yield self.fn(batch)


class MinigridPreprocess:

    def __init__(self, image_categorical=33, image_key='image', map_categorical=33, map_key='map', amp=False):
        self._image_categorical = image_categorical
        self._image_key = image_key
        self._map_categorical = map_categorical
        self._map_key = map_key
        self._amp = amp
        self._first = True

    def __call__(self, dataset: IterableDataset) -> IterableDataset:
        return TransformedDataset(dataset, self.apply)

    def apply(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self._first:
            print('Data batch (before preprocess): ', {k: v.shape for k, v in batch.items()})
            self._first = False

        # image

        batch['image'] = batch[self._image_key]  # Use something else (e.g. map_masked) as image
        if self._image_categorical:
            batch['image'] = to_onehot(batch['image'], self._image_categorical)
        else:
            batch['image'] = to_image(batch['image'])

        # map

        batch['map'] = batch[self._map_key]
        if self._map_categorical:
            batch['map'] = to_onehot(batch['map'], self._map_categorical)
        else:
            batch['map'] = to_image(batch['map'])

        # action

        assert len(batch['action'].shape) == 3  # should be already one-hot
        batch['action'] = batch['action'].astype(np.float32)

        # reward, terminal

        batch['reward'] = batch['reward'].astype(bool).astype(np.int64)  # assume reward 0 or 1
        batch['terminal'] = batch['terminal'].astype(np.int64)

        # map_coord

        if 'agent_pos' in batch and 'agent_dir' in batch:
            agent_pos = batch['agent_pos'] / 4.5 - 1.0  # TODO: make generic for any size
            agent_dir = batch['agent_dir']
            batch['map_coord'] = np.concatenate([agent_pos, agent_dir], axis=-1).astype(np.float32)

        # => float16
            
        if self._amp:
            batch['image'] = batch['image'].astype(np.float16)
            batch['map'] = batch['map'].astype(np.float16)
            batch['action'] = batch['action'].astype(np.float16)
            batch['map_coord'] = batch['map_coord'].astype(np.float16)
        return batch
