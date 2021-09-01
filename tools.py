from typing import Union
import warnings
from mlflow.tracking.client import MlflowClient
import yaml
import tempfile
from pathlib import Path
from pathy import Pathy
import io
import time
import numpy as np
import mlflow
import torch

warnings.filterwarnings("ignore", "Your application has authenticated using end user credentials")

print_once_keys = set()

def print_once(key, obj):
    if key not in print_once_keys:
        print_once_keys.add(key)
        print(key, obj)


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
    if resume_id:
        runs = mlflow.search_runs(filter_string=f'tags.resume_id="{resume_id}"')
        if len(runs) > 0:
            run_id = runs.run_id.iloc[0]
            print(f'Mlflow resuming run {run_id} ({resume_id})')
    run = mlflow.start_run(run_name=run_name, run_id=run_id, tags={'resume_id': resume_id or ''})
    print(f'Mlflow run {run.info.run_id} in experiment {run.info.experiment_id}')


def mlflow_log_npz(data, name, subdir=None, verbose=False):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / name
        save_npz(data, path)
        if verbose:
            print(f'Uploading artifact {subdir}/{name} size {path.stat().st_size/1024/1024:.2f} MB')
        mlflow.log_artifact(str(path), artifact_path=subdir)

def mlflow_log_text(text, name: str, subdir=None):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / name
        path.write_text(text)
        mlflow.log_artifact(str(path), artifact_path=subdir)

def mlflow_save_checkpoint(model, optimizer_wm, optimizer_map, optimizer_actor, optimizer_critic, steps):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / 'latest.pt'
        torch.save({
            'epoch': steps,
            'model_state_dict': model.state_dict(),
            'optimizer_wm_state_dict': optimizer_wm.state_dict(),
            'optimizer_map_state_dict': optimizer_map.state_dict(),
            'optimizer_actor_state_dict': optimizer_actor.state_dict(),
            'optimizer_critic_state_dict': optimizer_critic.state_dict(),
        }, path)
        mlflow.log_artifact(str(path), artifact_path='checkpoints')

def mlflow_load_checkpoint(model, optimizer_wm=None, optimizer_map=None, optimizer_actor=None, optimizer_critic=None, artifact_path='checkpoints/latest.pt', map_location=None):
    with tempfile.TemporaryDirectory() as tmpdir:
        client = MlflowClient()
        run_id = mlflow.active_run().info.run_id  # type: ignore
        try:
            path = client.download_artifacts(run_id, artifact_path, tmpdir)
        except:
            # Checkpoint not found
            return None
        checkpoint = torch.load(path, map_location=map_location)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer_wm:
            optimizer_wm.load_state_dict(checkpoint['optimizer_wm_state_dict'])
        if optimizer_map:
            optimizer_map.load_state_dict(checkpoint['optimizer_map_state_dict'])
        if optimizer_actor:
            optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        if optimizer_critic:
            optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
        return checkpoint['epoch']

def save_npz(data, filename):
    with io.BytesIO() as f1:
        np.savez_compressed(f1, **data)
        f1.seek(0)
        with filename.open('wb') as f2:
            f2.write(f1.read())

def load_npz(path: Union[Path, Pathy]):
    with io.BytesIO() as f1:
        with path.open('rb') as f:
            # For remote file it's faster to copy to memory buffer first
            f1.write(f.read())  
        f1.seek(0)
        fdata = np.load(f1)
        data = {key: fdata[key] for key in fdata}
    return data

def param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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