from typing import Callable, Dict, Tuple

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from .tools import *


def to_onehot(x: np.ndarray, n_categories) -> np.ndarray:
    e = np.eye(n_categories, dtype=np.float32)
    return e[x]  # Nice trick: https://stackoverflow.com/a/37323404


def img_to_onehot(x: np.ndarray, n_categories) -> np.ndarray:
    x = to_onehot(x, n_categories)
    x = x.transpose(0, 1, 4, 2, 3)  # (T, B, H, W, C) => (T, B, C, H, W)
    return x


def to_image(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        x = x.astype(np.float32)
        x = x / 255.0 - 0.5  # type: ignore
    else:
        assert 0.0 <= x[0, 0, 0, 0, 0] and x[0, 0, 0, 0, 0] <= 1.0
        x = x.astype(np.float32)
    x = x.transpose(0, 1, 4, 2, 3)  # (T, B, H, W, C) => (T, B, C, H, W)
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
            # Important: randomize global random seeds for spawned data generators.
            # Otherwise they share the same global random state.
            np.random.seed(np.random.RandomState().randint(999999))
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

    def __init__(self,
                 image_key='image', 
                 map_key=None, 
                 image_categorical=None, 
                 map_categorical=None, 
                 action_dim=0,
                 clip_rewards=None,
                 amp=False):
        self.image_categorical = image_categorical
        self.image_key = image_key
        self.map_categorical = map_categorical
        self.map_key = map_key
        self.action_dim = action_dim
        self.clip_rewards = clip_rewards
        self.amp = amp

    def __call__(self, dataset: IterableDataset) -> IterableDataset:
        return TransformedDataset(dataset, self.apply)

    def apply(self, batch: Dict[str, np.ndarray], expandTB=False) -> Dict[str, np.ndarray]:
        print_once('Preprocess batch (before): ', {k: v.shape + (v.dtype.name,) for k, v in batch.items()})

        # expand

        if expandTB:
            batch = {k: v[np.newaxis, np.newaxis] for k, v in batch.items()}  # (*) => (T=1,B=1,*)

        # cleanup policy info logged by actor, not to be confused with current values
        remove_keys(batch, ['policy_value', 'policy_entropy', 'action_prob'])

        T, B = batch['reward'].shape[:2]

        # image

        if self.image_key:
            batch['image'] = batch[self.image_key]  # Use something else (e.g. map_masked) as image
            if self.image_categorical:
                batch['image'] = img_to_onehot(batch['image'], self.image_categorical)
            else:
                batch['image'] = to_image(batch['image'])

        # map

        if self.map_key:
            batch['map'] = batch[self.map_key]
            if self.map_categorical:
                batch['map'] = img_to_onehot(batch['map'], self.map_categorical)
            else:
                batch['map'] = to_image(batch['map'])
            # cleanup unused
            remove_keys(batch, ['map_centered'])

        if 'map_seen' in batch:
            # map_seen contains 0 where map is unseen, otherwise =map
            batch['map_seen_mask'] = (batch['map_seen'] > 0).astype(int)  # type: ignore
            del batch['map_seen']
        elif 'map_vis' in batch:
            # map_vis shows how long ago cell was seen, if never, then equals to max_steps=500
            batch['map_seen_mask'] = (batch['map_vis'] < 500).astype(int)  # type: ignore
            del batch['map_vis']

        # action

        if len(batch['action'].shape) == 2:
            batch['action'] = to_onehot(batch['action'], self.action_dim)
        assert len(batch['action'].shape) == 3
        batch['action'] = batch['action'].astype(np.float32)

        # reward, terminal

        batch['terminal'] = batch.get('terminal', np.zeros((T, B))).astype(np.float32)
        batch['reward'] = batch.get('reward', np.zeros((T, B))).astype(np.float32)
        if self.clip_rewards == 'tanh':
            batch['reward'] = np.tanh(batch['reward'])  # type: ignore
        if self.clip_rewards == 'log1p':
            batch['reward'] = np.log1p(batch['reward'])  # type: ignore

        # map_coord

        if 'agent_pos' in batch and 'agent_dir' in batch:
            map_size = float(batch['map'].shape[-2])
            agent_pos = batch['agent_pos'] / map_size * 2 - 1.0
            agent_dir = batch['agent_dir']
            batch['map_coord'] = np.concatenate([agent_pos, agent_dir], axis=-1).astype(np.float32)  # type: ignore

        # vecobs

        if 'vecobs' in batch:
            batch['vecobs'] = batch['vecobs'].astype(np.float32)
        else:
            batch['vecobs'] = np.zeros((T, B, 64), np.float32)

        # => float16

        if self.amp:
            for key in ['image', 'action', 'map', 'map_coord', 'vecobs']:
                if key in batch:
                    batch[key] = batch[key].astype(np.float16)

        print_once('Preprocess batch (after): ', {k: v.shape + (v.dtype.name,) for k, v in batch.items()})
        return batch
