import yaml
import tempfile
from pathlib import Path
import io
import numpy as np
import mlflow


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