import warnings
import yaml
import tempfile
from pathlib import Path
import io
import time
import numpy as np
import mlflow

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
    run = mlflow.start_run(run_name=run_name, run_id=run_id, tags={'resume_id': resume_id})
    print(f'Mlflow run {run.info.run_id} in experiment {run.info.experiment_id}')


def mlflow_log_npz(data, name, subdir=None):
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / name
        save_npz(data, filepath)
        mlflow.log_artifact(str(filepath), artifact_path=subdir)


def save_npz(data, filename):
    with io.BytesIO() as f1:
        np.savez_compressed(f1, **data)
        f1.seek(0)
        with filename.open('wb') as f2:
            f2.write(f1.read())


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
        dt = time.time() - self.start_time
        # self.times.append(dt)
        self.start_time = None
        if self.verbose:
            self.debug_print(dt)

    def debug_print(self, dt):
        print(f'{self.name:<10}: {int(dt*1000):>5} ms')
