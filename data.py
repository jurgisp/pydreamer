import numpy as np
import random
import time
from pathlib import Path
from pathy import Pathy
from torch.utils.data import IterableDataset, get_worker_info

from tools import *
from modules_tools import *
from generator import parse_episode_name


def get_worker_id():
    worker_info = get_worker_info()
    worker_id = worker_info.id if worker_info else 0
    return worker_id


class OfflineDataSequential(IterableDataset):
    """Offline data which processes episodes sequentially"""

    def __init__(self, input_dir: Union[str, List[str]], batch_length, batch_size, skip_first=True, reload_interval=0, buffer_size=0, reset_interval=0):
        super().__init__()
        self.input_dirs = [input_dir] if isinstance(input_dir, str) else input_dir
        self.batch_length = batch_length
        self.batch_size = batch_size
        self.skip_first = skip_first
        self.reload_interval = reload_interval
        self.buffer_size = buffer_size
        self.reset_interval = reset_interval
        self._reload_files(True)
        assert len(self._files) > 0, 'No data found'

    def _reload_files(self, is_first=False):
        verbose = get_worker_id() == 0
        if is_first and verbose:
            print(f'Reading files from {self.input_dirs}...')

        files = []
        for dir in self.input_dirs:
            path = Pathy(dir) if dir.startswith('gs:/') or dir.startswith('s3:/') else Path(dir)
            files.extend(list(path.glob('*.npz')))
        files.sort()
        files_parsed = [(f, parse_episode_name(f.name)) for f in files]
        files_parsed.sort(key=lambda f__seed_ep_steps: -f__seed_ep_steps[1][1])  # Sort by episode number

        files_filtered = []
        steps_total = 0
        steps_filtered = 0
        for f, (seed, ep, steps, rew) in files_parsed:
            steps_total += steps
            if steps_total < self.buffer_size or not self.buffer_size:
                files_filtered.append(f)
                steps_filtered = steps_total

        self._files = files_filtered
        self._last_reload = time.time()
        self.stats_steps = steps_total

        if verbose:
            print(f'[TRAIN]  Found total files|steps: {len(files)}|{steps_total}, filtered: {len(self._files)}|{steps_filtered}')

    def _should_reload_files(self):
        return self.reload_interval and (time.time() - self._last_reload > self.reload_interval)

    def __iter__(self):
        # Parallel iteration over (batch_size) iterators
        # Iterates forever
        iters = [self._iter_single(ix) for ix in range(self.batch_size)]
        for batches in zip(*iters):
            batch = stack_structure_np(batches)
            batch = map_structure_np(batch, lambda d: d.swapaxes(0, 1))
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
                batch = cat_structure_np([last_partial_batch, batch])  # type: ignore
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
                data: Dict[str, np.ndarray] = load_npz(file)  # type: ignore
        except Exception as e:
            print('Error reading file - skipping')
            print(e)
            return

        # Undo the transformation for better compression
        if 'image' not in data and 'image_t' in data:
            data['image'] = data['image_t'].transpose(3, 0, 1, 2)  # HWCN => NHWC
            del data['image_t']

        if 'map_centered' in data and data['map_centered'].dtype == np.float64:
            assert False, 'Legacy, shouldnt happen anymore'  # TODO: remove
            # data['map_centered'] = (data['map_centered'] * 255).clip(0, 255).astype(np.uint8)

        # # Convert one-hot back to categorical
        # if len(data['action'].shape) == 2:
        #     data['action'] = data['action'].argmax(-1)

        n = lenb(data)
        if n < batch_length:
            print(f'Skipping too short file: {file}, len={n}')
            return

        if not 'reset' in data:
            data['reset'] = np.zeros(n, bool)
            data['reset'][0] = True  # Indicate episode start

        i = 0 if not skip_random else np.random.randint(n - batch_length + 1)
        l = first_shorter_length or batch_length

        if self.reset_interval:
            random_resets = self.randomize_resets(data['reset'], self.reset_interval)
        else:
            random_resets = np.zeros_like(data['reset'])

        while i < n:
            batch = {key: data[key][i:i + l] for key in data}
            if np.any(random_resets[i:i + l]):
                # Random resets are generated at any step, but always reset in the beginning of the batch, for longer backprop
                assert not np.any(batch['reset']), 'randomize_resets should not coincide with actual resets'
                batch['reset'][0] = True
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

    def randomize_resets(self, resets, reset_interval):
        assert resets[0]
        ep_boundaries = np.where(resets)[0].tolist() + [len(resets)]

        random_resets: np.ndarray = np.zeros_like(resets)  # type: ignore
        for i in range(len(ep_boundaries) - 1):
            ep_start = ep_boundaries[i]
            ep_end = ep_boundaries[i + 1]
            ep_steps = ep_end - ep_start

            # Cut episode into a random number of intervals

            max_intervals = (ep_steps // reset_interval) + 1
            n_intervals = np.random.randint(1, max_intervals + 1)
            i_boundaries = np.sort(np.random.choice(ep_steps - 50 * n_intervals, n_intervals - 1))
            i_boundaries = ep_start + i_boundaries + np.arange(1, n_intervals) * 50

            random_resets[i_boundaries] = True
            assert (resets | random_resets)[ep_start:ep_end].sum() == n_intervals

        return random_resets


def lenb(batch):
    return batch['reward'].shape[0]
