import pathlib
import numpy as np
import random


class OfflineData:

    def __init__(self, input_dir):
        input_dir = pathlib.Path(input_dir)
        self._files = list(sorted(input_dir.glob('*.npz')))
        print(f'Offline data: {len(self._files)} episodes in {str(input_dir)}')

    def iterate(self, batch_length, batch_size):
        # TODO: how many times should we iterate?
        # TODO: should offset single iterators by random amount, so episode starts are not synchronized

        # Parallel iteration over (batch_size) iterators
        iters = [self._iter_single(batch_length) for _ in range(batch_size)]
        for batches in zip(*iters):
            batch = {}
            for key in batches[0]:
                batch[key] = np.stack([b[key] for b in batches]).swapaxes(0, 1)
            yield batch

    def _iter_single(self, batch_length):
        # TODO: join files so we don't miss the last step indicating done

        for file in self._shuffle_files():
            for batch in self._iter_file(file, batch_length):
                yield batch

    def _iter_file(self, file, batch_length):
        with file.open('rb') as f:
            fdata = np.load(f)
            data = {key: fdata[key] for key in fdata}

        n = data['image'].shape[0]
        for i in range(0, n - batch_length + 1, batch_length):
            # TODO: should return last shorter batch
            j = i + batch_length
            batch = {key: data[key][i:j] for key in data}
            yield batch

    def _shuffle_files(self):
        files = self._files.copy()
        random.shuffle(files)
        return files
