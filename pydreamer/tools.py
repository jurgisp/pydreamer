import io
import logging
import os
import posixpath
import tempfile
import time
import warnings
from logging import debug, info
from pathlib import Path
from typing import Dict, Tuple, Union

import mlflow
import numpy as np
import yaml
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.store.artifact.azure_blob_artifact_repo import \
    AzureBlobArtifactRepository
from mlflow.tracking.client import MlflowClient

warnings.filterwarnings("ignore", "Your application has authenticated using end user credentials")

print_once_keys = set()


def print_once(key, obj):
    if key not in print_once_keys:
        print_once_keys.add(key)
        logging.debug(f'{key} {obj}')


def to_list(s):
    return s if isinstance(s, list) else [s]


def read_yamls(dir):
    conf = {}
    no_conf = True
    for config_file in Path(dir).glob('*.yaml'):
        no_conf = False
        with config_file.open('r') as f:
            conf.update(yaml.safe_load(f))
    if no_conf:
        print(f'WARNING: No yaml files found in {dir}')
    return conf


def mlflow_start_or_resume(run_name, resume_id=None):
    run_id = None
    info(f'Starting or resuming mlflow run ({os.environ.get("MLFLOW_TRACKING_URI", "local")}) ...')
    if resume_id:
        runs = mlflow.search_runs(filter_string=f'tags.resume_id="{resume_id}"')
        if len(runs) > 0:
            run_id = runs.run_id.iloc[0]
            info(f'Resumed mlflow run {run_id} ({resume_id})')
    run = mlflow.start_run(run_name=run_name, run_id=run_id, tags={'resume_id': resume_id or ''})
    info(f'Started mlflow run {run.info.run_id} in experiment {run.info.experiment_id}')
    return run


def mlflow_log_npz(data: dict, name, subdir=None, verbose=False, repository: ArtifactRepository = None):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / name
        save_npz(data, path)
        if verbose:
            print(f'Uploading artifact {subdir}/{name} size {path.stat().st_size/1024/1024:.2f} MB')
        if repository:
            repository.log_artifact(str(path), artifact_path=subdir)
        else:
            mlflow.log_artifact(str(path), artifact_path=subdir)


def mlflow_load_npz(name, repository: ArtifactRepository):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = Path(tmpdir) / name
        repository._download_file(name, tmpfile)  # TODO: avoid writing to disk - make sure tmp is RAM disk?
        return load_npz(tmpfile)


def mlflow_log_text(text, name: str, subdir=None):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / name
        path.write_text(text)
        mlflow.log_artifact(str(path), artifact_path=subdir)


def mlflow_save_checkpoint(model, optimizers, steps):
    import torch
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / 'latest.pt'
        checkpoint = {}
        checkpoint['epoch'] = steps
        checkpoint['model_state_dict'] = model.state_dict()
        for i, opt in enumerate(optimizers):
            checkpoint[f'optimizer_{i}_state_dict'] = opt.state_dict()
        torch.save(checkpoint, path)
        mlflow.log_artifact(str(path), artifact_path='checkpoints')


def mlflow_load_checkpoint(model, optimizers=tuple(), artifact_path='checkpoints/latest.pt', map_location=None):
    import torch
    with tempfile.TemporaryDirectory() as tmpdir:
        client = MlflowClient()
        run_id = mlflow.active_run().info.run_id  # type: ignore
        try:
            path = client.download_artifacts(run_id, artifact_path, tmpdir)
        except Exception as e:  # TODO: check if it's an error instead of expected "not found"
            # Checkpoint not found
            return None
        checkpoint = torch.load(path, map_location=map_location)
        model.load_state_dict(checkpoint['model_state_dict'])
        for i, opt in enumerate(optimizers):
            opt.load_state_dict(checkpoint[f'optimizer_{i}_state_dict'])
        return checkpoint['epoch']


def save_npz(data, path):
    if isinstance(path, str):
        path = Path(path)
    with io.BytesIO() as f1:
        np.savez_compressed(f1, **data)  # Save to memory buffer first ...
        f1.seek(0)
        with path.open('wb') as f2:
            f2.write(f1.read())  # ... then write it to file


def load_npz(path, keys=None) -> Dict[str, np.ndarray]:
    if isinstance(path, str):
        path = Path(path)
    with path.open('rb') as f:
        fdata: Dict[str, np.ndarray] = np.load(f)  # type: ignore
        if keys is None:
            data = {key: fdata[key] for key in fdata}
        else:
            data = {key: fdata[key] for key in keys}
    return data


def param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def discount(x: np.ndarray, gamma: float) -> np.ndarray:
    import scipy.signal
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]  # type: ignore


class Timer:

    def __init__(self, name='timer', verbose=True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        # self.times = []

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.dt = time.time() - self.start_time  # type: ignore
        # self.times.append(dt)
        self.start_time = None
        if self.verbose:
            self.debug_print(self.dt)

    def debug_print(self, dt):
        print(f'{self.name:<10}: {int(dt*1000):>5} ms')

    @property
    def dt_ms(self):
        return int(self.dt * 1000)


class NoProfiler:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def step(self):
        pass


def azure_blob_artifact_repo_log_artifact(self, local_file, artifact_path=None):
    (container, _, dest_path) = self.parse_wasbs_uri(self.artifact_uri)
    container_client = self.client.get_container_client(container)
    if artifact_path:
        dest_path = posixpath.join(dest_path, artifact_path)
    dest_path = posixpath.join(dest_path, os.path.basename(local_file))
    with open(local_file, "rb") as file:
        # container_client.upload_blob(dest_path, file)  # original
        container_client.upload_blob(dest_path, file, overwrite=True)  # patched


# Patching to enable artifact overwrite when using Azure, which is default in GCS
#   https://github.com/mlflow/mlflow/blob/master/mlflow/store/artifact/azure_blob_artifact_repo.py#L75
AzureBlobArtifactRepository.log_artifact = azure_blob_artifact_repo_log_artifact


def chunk_episode_data(data: Dict[str, np.ndarray], min_length: int):
    n = len(data['reward'])
    chunks = n // min_length
    for i_chunk in range(chunks):
        i_from = n * i_chunk // chunks
        i_to = n * (i_chunk + 1) // chunks
        data_chunk = {key: data[key][i_from:i_to] for key in data}
        yield data_chunk


class LogColorFormatter(logging.Formatter):
    # see https://stackoverflow.com/questions/4842424/list-of-ansi-color-escape-sequences
    GREY = '\033[90m'
    WHITE = '\033[37m'
    GREEN = '\033[32m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RED_UNDERLINE = '\033[4;91m'

    def __init__(self,
                 fmt,
                 debug_color=GREY,
                 info_color=None,
                 warning_color=YELLOW,
                 error_color=RED,
                 critical_color=RED_UNDERLINE
                 ):
        super().__init__(fmt)
        self.fmt = fmt
        self.debug_color = debug_color
        self.info_color = info_color
        self.warning_color = warning_color
        self.error_color = error_color
        self.critical_color = critical_color

    def format(self, record):
        RESET = '\033[0m'
        if record.levelno == logging.DEBUG:
            fmt = f'{self.debug_color or ""}{self.fmt}{RESET}'
        elif record.levelno == logging.INFO:
            fmt = f'{self.info_color or ""}{self.fmt}{RESET}'
        elif record.levelno == logging.WARNING:
            fmt = f'{self.warning_color or ""}{self.fmt}{RESET}'
        elif record.levelno == logging.ERROR:
            fmt = f'{self.error_color or ""}{self.fmt}{RESET}'
        elif record.levelno == logging.CRITICAL:
            fmt = f'{self.critical_color or ""}{self.fmt}{RESET}'
        else:
            fmt = self.fmt
        return logging.Formatter(fmt).format(record)
