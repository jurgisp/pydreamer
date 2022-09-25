import io
import logging
import os
import posixpath
import sys
import tempfile
import time
import warnings
from logging import debug, info, exception
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import yaml

try:
    from mlflow.store.artifact.artifact_repo import ArtifactRepository
except:
    ArtifactRepository = Any  # just for type annotation

# Ignore Google Cloud Storage warnings
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
    for config_file in Path(dir).glob('**/*.yaml'):
        no_conf = False
        with config_file.open('r') as f:
            conf.update(yaml.safe_load(f))
    if no_conf:
        print(f'WARNING: No yaml files found in {dir}')
    return conf


def mlflow_init(wait_for_resume=False):
    import mlflow
    run_name = os.environ.get('MLFLOW_RUN_NAME')
    resume_id = os.environ.get('MLFLOW_RESUME_ID')
    uri = os.environ.get('MLFLOW_TRACKING_URI', 'local')

    run = mlflow.active_run()
    if run:
        # Run already active
        pass

    elif os.environ.get('MLFLOW_RUN_ID'):
        # Run not active, but specific ID set (probably subprocess)
        run = mlflow.start_run(run_id=os.environ['MLFLOW_RUN_ID'])
        info(f'Reinitialized mlflow run {run.info.run_id} ({resume_id}) in {uri}/{run.info.experiment_id}')

    else:
        resume_run_id = None
        if resume_id:
            # Resume ID specified - try to find the same run
            while True:
                runs = mlflow.search_runs(filter_string=f'tags.resume_id="{resume_id}"')
                if len(runs) > 0:
                    resume_run_id = runs.run_id.iloc[0]  # type: ignore
                    break
                else:
                    if wait_for_resume:
                        debug(f'Waiting until mlflow run ({resume_id}) is available...')
                        time.sleep(10)
                    else:
                        break
        else:
            assert not wait_for_resume, "Wait for resume, but no MLFLOW_RESUME_ID"

        if resume_run_id:
            # Resuming run
            run = mlflow.start_run(run_id=resume_run_id)
            info(f'Resumed mlflow run {run.info.run_id} ({resume_id}) in {uri}/{run.info.experiment_id}')
        else:
            # Starting new run
            run = mlflow.start_run(run_name=run_name, tags={'resume_id': resume_id or ''})
            info(f'Started mlflow run {run.info.run_id} ({resume_id}) in {uri}/{run.info.experiment_id}')

    os.environ['MLFLOW_RUN_ID'] = run.info.run_id  # for subprocesses
    return run


def mlflow_log_params(params: dict):
    import mlflow
    MAX_VALUE_LENGTH = 250
    MAX_BATCH_SIZE = 100
    kvs = [
        (str(k), str(v))
        for k, v in params.items()
        if 1 <= len(str(v)) <= MAX_VALUE_LENGTH  # Filter out too long values. Also filter out empty strings, for some reason causes INVALID_PARAMETER_VALUE
    ]
    for i in range(0, len(kvs), MAX_BATCH_SIZE):  # log_params() allows max 100 items
        try:
            params_batch = dict(kvs[i:i + MAX_BATCH_SIZE])
            mlflow.log_params(params_batch)
        except Exception as e:
            # This happens when resuming and config has different parameters - it's fine
            exception('Error in mlflow.log_params (it is ok if params changed).')


def mlflow_log_metrics(metrics: dict, step: int):
    import mlflow
    while True:
        try:
            mlflow.log_metrics(metrics, step=step)
            break
        except:
            exception('Error logging metrics - will retry.')
            time.sleep(10)


def mlflow_log_npz(data: dict, name, subdir=None, verbose=False, repository: ArtifactRepository = None):
    import mlflow
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / name
        save_npz(data, path)
        mlflow_log_artifact(path, subdir, verbose, repository)


def mlflow_log_artifact(path: Path, subdir=None, verbose=True, repository: ArtifactRepository = None):
    import mlflow
    if verbose:
        debug(f'Uploading artifact {subdir}/{path.name} size {path.stat().st_size/1024/1024:.2f} MB')
    while True:
        try:
            if repository:
                repository.log_artifact(str(path), artifact_path=subdir)
            else:
                mlflow.log_artifact(str(path), artifact_path=subdir)
            break
        except:
            exception('Error saving artifact - will retry.')
            time.sleep(10)


def mlflow_load_npz(name, repository: ArtifactRepository):
    import mlflow
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = Path(tmpdir) / name
        repository._download_file(name, tmpfile)  # TODO: avoid writing to disk - make sure tmp is RAM disk?
        return load_npz(tmpfile)


def mlflow_log_text(text, name: str, subdir=None):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / name
        path.write_text(text)
        mlflow_log_artifact(path, subdir)


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
        mlflow_log_artifact(path, subdir='checkpoints')


def mlflow_load_checkpoint(model, optimizers=tuple(), artifact_path='checkpoints/latest.pt', map_location=None):
    import mlflow
    from mlflow.tracking.client import MlflowClient
    import torch
    with tempfile.TemporaryDirectory() as tmpdir:
        client = MlflowClient()
        run_id = mlflow.active_run().info.run_id  # type: ignore
        try:
            path = client.download_artifacts(run_id, artifact_path, tmpdir)
        except Exception as e:  # TODO: check if it's an error instead of expected "not found"
            # Checkpoint not found
            return None
        try:
            checkpoint = torch.load(path, map_location=map_location)
        except:
            exception('Error reading checkpoint')
            return None
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
        fdata: Dict[str, np.ndarray] = np.load(f)
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


def chunk_episode_data(data: Dict[str, np.ndarray], min_length: int):
    # this n=len(..)-1 and i_to=(...)+1 makes first reset step "free" and correctly chunks 2000 episode into (1001)+(1000)
    n = len(data['reward']) - 1
    chunks = n // min_length
    i_from = 0
    for i_chunk in range(chunks):
        i_to = n * (i_chunk + 1) // chunks + 1
        data_chunk = {key: data[key][i_from:i_to] for key in data}
        i_from = i_to
        yield data_chunk


class LogColorFormatter(logging.Formatter):
    # see https://stackoverflow.com/questions/4842424/list-of-ansi-color-escape-sequences
    GREY = '\033[90m'
    WHITE = '\033[37m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    RED = '\033[31m'
    RED_UNDERLINE = '\033[4;31m'

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


def configure_logging(prefix='[%(name)s]', level=logging.DEBUG, info_color=None):
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(LogColorFormatter(
        f'{prefix}  %(message)s',
        info_color=info_color
    ))
    logging.root.setLevel(level)
    logging.root.handlers = [handler]
    for logname in ['urllib3', 'requests', 'mlflow', 'git', 'azure', 'PIL', 'numba', 'google.auth']:
        logging.getLogger(logname).setLevel(logging.WARNING)  # disable other loggers
    for logname in ['absl', 'minerl']:
        logging.getLogger(logname).setLevel(logging.INFO)


def tensorboard_trace_handler(dir_name: str, worker_name: Optional[str] = None, use_gzip: bool = False):
    """Forked from: torch.profiler.profiler.tensorboard_trace_handler.

    Outputs tracing files to directory of ``dir_name``, then that directory can be
    directly delivered to tensorboard as logdir.
    ``worker_name`` should be unique for each worker in distributed scenario,
    it will be set to '[hostname]_[pid]' by default.
    """
    import os
    import socket
    import time
    import mlflow

    def handler_fn(prof) -> None:
        nonlocal worker_name
        if not os.path.isdir(dir_name):
            try:
                os.makedirs(dir_name, exist_ok=True)
            except Exception:
                raise RuntimeError("Can't create directory: " + dir_name)
        if not worker_name:
            worker_name = "{}_{}".format(socket.gethostname(), str(os.getpid()))
        file_name = "{}.{}.pt.trace.json".format(worker_name, int(time.time() * 1000))
        if use_gzip:
            file_name = file_name + '.gz'
        path = os.path.join(dir_name, file_name)
        prof.export_chrome_trace(path)
        # PATCH: upload to mlflow
        debug(f'Uploading artifact {path} size {Path(path).stat().st_size/1024/1024:.2f} MB')
        mlflow.log_artifact(path, artifact_path='profiling')

    return handler_fn
