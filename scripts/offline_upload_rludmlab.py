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

import argparse
import io
import os
import tempfile
from datetime import datetime
from pathlib import Path
from PIL import Image

import mlflow
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tools import *
# from generator import build_episode_name, parse_episode_name


args = argparse.ArgumentParser()
args.add_argument('--env', default='explore_object_rewards_few')
args.add_argument('--shard_from', default=0, type=int)
args.add_argument('--shard_to', default=1500, type=int)
args.add_argument('--resume_id', default='')

H, W, A = 72, 96, 15
STEPS_PER_NPZ = 1800


def build_episode_name(seed, episode_from, episode, reward, steps):
    if episode_from < episode:
        return f's{seed}-ep{episode_from:06}_{episode:06}-r{reward}-{steps:04}.npz'
    else:
        return f's{seed}-ep{episode:06}-r{reward}-{steps:04}.npz'


def parse_episode_name(fname):
    # fname = 's{seed}-ep{epfrom}_{episode}-r{reward}-{steps}.npz'
    #       | 's{seed}-ep{episode}-r{reward}-{steps}.npz'
    seed = fname.split('-')[0][1:]
    seed = int(seed) if seed.isnumeric() else 0
    steps = fname.split('.')[0].split('-')[-1]
    steps = int(steps) if steps.isnumeric() else 0
    episode = fname.split('.')[0].split('-')[-3].replace('ep', '').split('_')[-1]
    episode = int(episode) if episode.isnumeric() else 0
    reward = fname.split('-r')[-1].split('-')[0]  # Doen't handle negatives
    reward = int(reward) if reward.isnumeric() else 0
    return (seed, episode, steps, reward)


def decode_image(imgb, h=64, w=64):
    img = tf.io.decode_image(imgb)
    img = img[:, :, ::-1].numpy()  # BGR => RGB
    img = np.array(Image.fromarray(img).resize((64, 64), Image.NEAREST))  # resize
    return img


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


def check_remaining_shards(artifact_dir, seed_from, seed_to):
    files = list(sorted(artifact_dir.glob('*.npz')))
    max_seed = -1
    for f in files:
        fseed, fepisode, fsteps, _ = parse_episode_name(f.name)
        if not (fseed >= seed_from and fseed < seed_to):
            continue  # Belongs to another generator
        if fseed > max_seed:
            max_seed = fseed
    print(f'Found existing {len(files)} files in {artifact_dir}, max shard {max_seed} from range [{seed_from},{seed_to})')
    if max_seed >= 0:
        print(f'Will continue from {max_seed}...')
    return max(max_seed, seed_from), seed_to


if __name__ == '__main__':
    conf = args.parse_args()
    print(vars(conf))
    env = conf.env

    run = mlflow_start_or_resume(f'dmlab_{env}_{conf.shard_from}_{conf.shard_to}', conf.resume_id)

    episodes_dir = 'episodes'
    artifact_dir = run.info.artifact_uri.replace('file://', '') + '/' + episodes_dir
    if artifact_dir.startswith('gs:/') or artifact_dir.startswith('s3:/'):
        artifact_dir = Pathy(artifact_dir)
    else:
        artifact_dir = Path(artifact_dir)
    shard_from, shard_to = check_remaining_shards(artifact_dir, conf.shard_from, conf.shard_to)

    if env == 'rooms_watermaze':
        rluds = tfds.rl_unplugged.RluDmlabRoomsWatermaze()
    elif env == 'rooms_select_nonmatching_object':
        rluds = tfds.rl_unplugged.RluDmlabRoomsSelectNonmatchingObject()
    elif env == 'explore_object_rewards_few':
        rluds = tfds.rl_unplugged.RluDmlabExploreObjectRewardsFew()
    elif env == 'explore_object_rewards_many':
        rluds = tfds.rl_unplugged.RluDmlabExploreObjectRewardsMany()
    else:
        assert False, env

    step_counter = 0
    for shard in range(shard_from, shard_to):
        gs_path = f'gs://rl_unplugged/dmlab/{env}/training_{shard // 500}/tfrecord-{(shard % 500):05}-of-00500'
        print(gs_path, '...')
        ds = tf.data.TFRecordDataset(gs_path, compression_type='GZIP')
        ds = ds.map(rluds.tf_example_to_step_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE)
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
                step_counter += datas_steps
                mlflow.log_metrics({'_step': step_counter, '_timestamp': datetime.now().timestamp()})
