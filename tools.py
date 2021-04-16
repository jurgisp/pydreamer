import warnings
from mlflow.tracking.client import MlflowClient
import yaml
import tempfile
from pathlib import Path
import io
import time
import numpy as np
import mlflow
import torch

warnings.filterwarnings("ignore", "Your application has authenticated using end user credentials")


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


def mlflow_log_npz(data, name, subdir=None):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / name
        save_npz(data, path)
        mlflow.log_artifact(str(path), artifact_path=subdir)


def mlflow_save_checkpoint(model, optimizer, steps):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / 'latest.pt'
        torch.save({
            'epoch': steps,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
        mlflow.log_artifact(str(path), artifact_path='checkpoints')

def mlflow_load_checkpoint(model, optimizer, artifact_path = 'checkpoints/latest.pt'):
    with tempfile.TemporaryDirectory() as tmpdir:
        client = MlflowClient()
        run_id = mlflow.active_run().info.run_id
        try:
            path = client.download_artifacts(run_id, artifact_path, tmpdir)
        except:
            # Checkpoint not found
            return None
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']

def save_npz(data, filename):
    with io.BytesIO() as f1:
        np.savez_compressed(f1, **data)
        f1.seek(0)
        with filename.open('wb') as f2:
            f2.write(f1.read())


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
        dt = time.time() - self.start_time  # type: ignore
        # self.times.append(dt)
        self.start_time = None
        if self.verbose:
            self.debug_print(dt)

    def debug_print(self, dt):
        print(f'{self.name:<10}: {int(dt*1000):>5} ms')
