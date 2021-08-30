import numpy as np
import random
import time
from pathlib import Path
from pathy import Pathy
from torch.utils.data import IterableDataset, get_worker_info

from tools import *
from modules_tools import *


def get_worker_id():
    worker_info = get_worker_info()
    worker_id = worker_info.id if worker_info else 0
    return worker_id


class OfflineDataSequential(IterableDataset):
    """Offline data which processes episodes sequentially"""

    def __init__(self, input_dir: str, batch_length, batch_size, skip_first=True, reload_interval=0):
        super().__init__()
        if input_dir.startswith('gs:/') or input_dir.startswith('s3:/'):
            self.input_dir = Pathy(input_dir)
        else:
            self.input_dir = Path(input_dir)
        self.batch_length = batch_length
        self.batch_size = batch_size
        self.skip_first = skip_first
        self.reload_interval = reload_interval
        self._reload_files(True)
        assert len(self._files) > 0, 'No data found'

    def _reload_files(self, is_first=False):
        verbose = get_worker_id() == 0
        if is_first and verbose:
            print(f'Reading files from {str(self.input_dir)}...')
        self._files = list(sorted(self.input_dir.glob('*.npz')))
        self._last_reload = time.time()
        steps = 0
        for f in self._files:
            s = f.name.split('.')[0].split('-')[-1]
            if s.isnumeric():
                steps += int(s)
        if verbose:
            print(f'[TRAIN]  Found {len(self._files)} files, {steps} steps')
        self.stats_steps = steps

    def _should_reload_files(self):
        return self.reload_interval and (time.time() - self._last_reload > self.reload_interval)

    def __iter__(self):
        # Parallel iteration over (batch_size) iterators
        # Iterates forever
        iters = [self._iter_single(ix) for ix in range(self.batch_size)]
        for batches in zip(*iters):
            batch = {}
            for key in batches[0]:
                batch[key] = np.stack([b[key] for b in batches]).swapaxes(0, 1)
            yield batch

    def _iter_single(self, ix):
        # Iterates "single thread" forever
        skip_random = self.skip_first
        last_partial_batch = None

        for file in self._iter_shuffled_files():
            if last_partial_batch:
                first_shorter_length = self.batch_length - lenb(last_partial_batch)
            else:
                first_shorter_length = None

            it = self._iter_file(file, self.batch_length, skip_random, first_shorter_length)

            # Concatenate the last batch of previous file and the first batch of new file to make a
            # full batch of length batch_size.
            if last_partial_batch is not None:
                batch, partial = next(it)
                assert not partial, 'First batch must be full. Is episode_length < batch_size?'
                batch = cat_structure_np([last_partial_batch, batch])
                assert lenb(batch) == self.batch_length
                last_partial_batch = None
                yield batch

            for batch, partial in it:
                if partial:
                    last_partial_batch = batch
                    break  # partial will always be last
                yield batch

            skip_random = False

    def _iter_file(self, file, batch_length, skip_random=False, first_shorter_length=None):
        try:
            with Timer(f'Reading {file}', verbose=False):
                data = load_npz(file)
        except Exception as e:
            print('Error reading file - skipping')
            print(e)
            return

        n = lenb(data)
        if n < batch_length:
            print(f'Skipping too short file: {file}, len={n}')
            return

        i = 0 if not skip_random else np.random.randint(n - batch_length + 1)
        l = first_shorter_length or batch_length

        # Undo the transformation for better compression
        if 'image' not in data and 'image_t' in data:
            data['image'] = data['image_t'].transpose(3, 0, 1, 2)  # HWCN => NHWC
            del data['image_t']

        if 'map_centered' in data and data['map_centered'].dtype == np.float64:
            data['map_centered'] = (data['map_centered'] * 255).clip(0, 255).astype(np.uint8)

        if not 'reset' in data:
            data['reset'] = np.zeros(n, bool)
            data['reset'][0] = True  # Indicate episode start

        while i < n:
            batch = {key: data[key][i:i + l] for key in data}
            is_partial = lenb(batch) < l
            i += l
            l = batch_length
            yield batch, is_partial

    def _iter_shuffled_files(self):
        while True:
            i = random.randint(0, len(self._files) - 1)
            f = self._files[i]
            if not f.exists() or self._should_reload_files():
                self._reload_files()
            else:
                yield self._files[i]


def lenb(batch):
    return batch['reward'].shape[0]
