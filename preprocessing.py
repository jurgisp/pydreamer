from typing import Dict, Tuple, Callable
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info
from tools import *


def to_onehot(x: np.ndarray, n_categories) -> np.ndarray:
    e = np.eye(n_categories, dtype=np.float32)
    return e[x]  # Nice trick: https://stackoverflow.com/a/37323404

def img_to_onehot(x: np.ndarray, n_categories) -> np.ndarray:
    x = to_onehot(x, n_categories)
    x = x.transpose(0, 1, 4, 2, 3)  # (N, B, H, W, C) => (N, B, C, H, W)
    return x


def to_image(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        x = x.astype(np.float32)
        x = x / 255.0 - 0.5  # type: ignore
    else:
        assert 0.0 <= x[0, 0, 0, 0, 0] and x[0, 0, 0, 0, 0] <= 1.0
        x = x.astype(np.float32)
    x = x.transpose(0, 1, 4, 2, 3)  # (N, B, H, W, C) => (N, B, C, H, W)
    return x


def remove_keys(data: dict, keys: list):
    for key in keys:
        if key in data:
            del data[key]


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


class Preprocessor:

    def __init__(self, image_categorical=33, image_key='image', map_categorical=33, map_key='map', action_dim=0, amp=False):
        self._image_categorical = image_categorical
        self._image_key = image_key
        self._map_categorical = map_categorical
        self._map_key = map_key
        self._action_dim = action_dim
        self._amp = amp

    def __call__(self, dataset: IterableDataset) -> IterableDataset:
        return TransformedDataset(dataset, self.apply)

    def apply(self, batch: Dict[str, np.ndarray], expandTB=False) -> Dict[str, np.ndarray]:
        print_once('Preprocess batch (before): ', {k: v.shape for k, v in batch.items()})

        # expand

        if expandTB:
            batch = {k: v[np.newaxis, np.newaxis] for k, v in batch.items()}  # (*) => (T=1,B=1,*)

        # cleanup policy info logged by actor, not to be confused with current values
        remove_keys(batch, ['policy_value', 'policy_entropy', 'action_prob'])

        # image

        batch['image'] = batch[self._image_key]  # Use something else (e.g. map_masked) as image
        T, B, C, H, W = batch['image'].shape
        if self._image_categorical:
            batch['image'] = img_to_onehot(batch['image'], self._image_categorical)
        else:
            batch['image'] = to_image(batch['image'])

        # map

        if self._map_key:
            batch['map'] = batch[self._map_key]
            if self._map_categorical:
                batch['map'] = img_to_onehot(batch['map'], self._map_categorical)
            else:
                batch['map'] = to_image(batch['map'])
            # cleanup unused
            remove_keys(batch, ['map_centered'])
        else:
            batch['map'] = np.zeros((T, B, 1, 1, 1))

        # action

        if len(batch['action'].shape) == 2:
            batch['action'] = to_onehot(batch['action'], self._action_dim)
        assert len(batch['action'].shape) == 3
        batch['action'] = batch['action'].astype(np.float32)

        # reward, terminal

        batch['reward'] = batch['reward'].astype(np.float32)
        batch['terminal'] = batch['terminal'].astype(np.float32)

        # map_coord

        if 'agent_pos' in batch and 'agent_dir' in batch:
            map_size = float(batch['map'].shape[-2])
            agent_pos = batch['agent_pos'] / map_size * 2 - 1.0
            agent_dir = batch['agent_dir']
            batch['map_coord'] = np.concatenate([agent_pos, agent_dir], axis=-1).astype(np.float32)  # type: ignore
        else:
            batch['map_coord'] = np.zeros((T, B, 4))

        # => float16

        if self._amp:
            for key in ['image', 'action', 'map', 'map_coord']:
                if key in batch:
                    batch[key] = batch[key].astype(np.float16)

        print_once('Preprocess batch (after): ', {k: v.shape for k, v in batch.items()})
        return batch
