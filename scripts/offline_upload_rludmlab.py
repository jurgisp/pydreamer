"""
## Create

gcloud compute instances create adhoc-jurgis-cpu-tf2 \
        --zone="europe-west4-b" \
        --image-family="tf2-latest-cpu-ubuntu-1804 " \
        --image-project="deeplearning-platform-release" \
        --maintenance-policy=TERMINATE \
        --machine-type="n1-standard-2" \
        --boot-disk-size="1200GB" \
        --subnet="private-vpc"
gcloud compute config-ssh
ssh adhoc-jurgis-cpu-tf2.europe-west4-b.human-ui
    pip3 install tfds-nightly
    pip3 install mlflow

    gcloud auth application-default login


## Start

gcloud compute instances start adhoc-jurgis-cpu-tf2
gcloud compute config-ssh
ssh adhoc-jurgis-cpu-tf2.europe-west4-b.human-ui
tmux
    python3


## Stop

gcloud compute instances stop adhoc-jurgis-cpu-tf2


## Connect

ssh adhoc-jurgis-cpu-tf2.europe-west4-b.human-ui
tmux attach

"""

import os
import io
import tempfile
from pathlib import Path
import mlflow
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# env = 'watermaze'
env = 'tmaze'

os.environ['MLFLOW_TRACKING_URI'] = 'http://10.164.0.62:30000'
os.environ['MLFLOW_EXPERIMENT_NAME'] = 'dreamer2_episodes'
mlflow.start_run(run_name=f'rlu_dmlab_{env}_0')

H, W, A = 72, 96, 15
STEPS_PER_NPZ = 1800

def decode_image(imgb, h=64, w=64):
    img = tf.io.decode_image(imgb)
    # img = tf.image.crop_to_bounding_box(img, (H-h)//2, (W-w)//2, h, w)  # center crop
    img = tf.image.crop_to_bounding_box(img, (H-h), (W-w)//2, h, w) # center-bottom crop
    img = img[:, :, ::-1]  # BGR => RGB?
    return img.numpy()

def parse_record(rec):
    # Saved data sample:
    # {'action': (2004, 3), 'reward': (2004,), 'terminal': (2004,), 'reset': (2004,), 'image_t': (64, 64, 3, 2004)}
    data = {}
    e = np.eye(A, dtype=np.float32)
    action = e[rec['steps']['observation']['last_action']]
    action[0] *= 0  # no zeroth action
    data['action'] = action
    data['reward'] = rec['steps']['observation']['last_reward']
    data['terminal'] = rec['steps']['is_terminal']
    data['reset'] = rec['steps']['is_first']
    data['image'] = np.stack([decode_image(imgb) for imgb in rec['steps']['observation']['pixels']], 0)
    return data

def build_episode_name(seed, episode_from, episode, reward, steps):
    if episode_from < episode:
        return f's{seed}-ep{episode_from:06}_{episode:06}-r{reward}-{steps:04}.npz'
    else:
        return f's{seed}-ep{episode:06}-r{reward}-{steps:04}.npz'

def mlflow_log_npz(data, name, subdir=None, verbose=False):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / name
        save_npz(data, path)
        if verbose:
            print(f'Uploading artifact {subdir}/{name} size {path.stat().st_size/1024/1024:.2f} MB')
        mlflow.log_artifact(str(path), artifact_path=subdir)

def save_npz(data, filename):
    with io.BytesIO() as f1:
        np.savez_compressed(f1, **data)
        f1.seek(0)
        with filename.open('wb') as f2:
            f2.write(f1.read())

if env == 'watermaze':
    rluds = tfds.rl_unplugged.RluDmlabRoomsWatermaze()
elif env == 'tmaze':
    rluds = tfds.rl_unplugged.RluDmlabRoomsSelectNonmatchingObject()
else:
    assert False, env

for shard in range(500):
    if env == 'watermaze':
        ds = tf.data.TFRecordDataset(f'gs://rl_unplugged/dmlab/rooms_watermaze/training_0/tfrecord-{shard:05}-of-00500', compression_type='GZIP')
    elif env == 'tmaze':
        ds = tf.data.TFRecordDataset(f'gs://rl_unplugged/dmlab/rooms_select_nonmatching_object/training_0/tfrecord-{shard:05}-of-00500', compression_type='GZIP')
    else:
        assert False, env
    ds = ds.map(rluds.tf_example_to_step_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    episodes_dir = 'episodes_eval' if (shard == 0 or shard > 450) else 'episodes'
    datas = []
    for i, r in enumerate(ds.as_numpy_iterator()):
        data = parse_record(r)
        if i == 0:
            print('Episode data sample: ', {k: v.shape for k, v in data.items()})
        #
        # Save to npz
        datas.append(data)
        datas_episodes = len(datas)
        datas_steps = sum(len(d['reset']) - 1 for d in datas)
        datas_reward = sum(d['reward'].sum() for d in datas)
        if datas_steps >= STEPS_PER_NPZ:
            # Concatenate episodes
            data = {}
            for key in datas[0]:
                data[key] = np.concatenate([b[key] for b in datas], axis=0)
            datas = []
            # NHWC => HWCN for better compression
            data['image_t'] = data['image'].transpose(1, 2, 3, 0)
            del data['image']
            # Save to npz
            if i <= datas_episodes:
                print('Saved data sample: ', {k: v.shape for k, v in data.items()})
            fname = build_episode_name(shard, i + 1 - datas_episodes, i, int(datas_reward), datas_steps)
            mlflow_log_npz(data, fname, episodes_dir, verbose=True)
