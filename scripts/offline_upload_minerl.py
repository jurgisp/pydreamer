import argparse
import io
import os
import sys
import tempfile
import warnings
from functools import partialmethod
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore", ".*Box bound precision lowered by casting")
warnings.filterwarnings("ignore", "Your application has authenticated using end user credentials")

import minerl
import mlflow
import numpy as np
from generator import build_episode_name
from sklearn.cluster import KMeans
from tools import *
from tqdm import tqdm

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # type: ignore


args = argparse.ArgumentParser()
args.add_argument('--run', default='minerl_treechop_obf')
args.add_argument('--env', default='MineRLTreechopVectorObf-v0')
args.add_argument('--cluster', default=False)

ACTION_REPEAT = 4
N_ACTIONS = 200
N_EPISODES_CLUSTER = 100
DATA_DIR = '/Users/jurgis/Documents/minerl-baselines/data'
ACTION_CENTROIDS_PATH = f'data/minerl_action_centroids_{ACTION_REPEAT}.npy'


def cluster_actions(env="MineRLObtainIronPickaxeVectorObf-v0"):
    dataset = minerl.data.make(env, data_dir=DATA_DIR, num_workers=1)  # type: ignore
    ep_names = dataset.get_trajectory_names()
    np.random.shuffle(ep_names)

    # Read all actions

    all_actions = []
    n = min(len(ep_names), N_EPISODES_CLUSTER)
    print(f'Collecting data from: {n} episodes')
    for i in range(n):
        print(i)
        episode = dataset.load_data(ep_names[i], skip_interval=0, include_metadata=False)
        curr_action = []
        for obs, act, _, _, _ in episode:
            curr_action.append(act["vector"])
            if len(curr_action) == ACTION_REPEAT:
                all_actions.append(np.array(curr_action).reshape(-1))
                curr_action = []

        if len(curr_action) > 0:
            while len(curr_action) < ACTION_REPEAT:
                curr_action.append(curr_action[-1])
            all_actions.append(np.array(curr_action).reshape(-1))

    all_actions = np.array(all_actions)

    print(f'Running KMeans: {all_actions.shape}')
    kmeans = KMeans(n_clusters=N_ACTIONS)
    kmeans.fit(all_actions)
    action_centroids: np.ndarray = kmeans.cluster_centers_  # type: ignore
    np.save(ACTION_CENTROIDS_PATH, action_centroids)
    print(f'Action centroids {action_centroids.shape} saved to: ', ACTION_CENTROIDS_PATH)


def save_episodes(conf):

    print(os.environ['MLFLOW_EXPERIMENT_NAME'])
    mlflow.start_run(run_name=conf.run)

    action_centroids = np.load(ACTION_CENTROIDS_PATH)
    mlflow_log_npz(dict(action_centroids=action_centroids), 'action_centroids.npz', verbose=True)

    dataset = minerl.data.make(conf.env, data_dir=DATA_DIR, num_workers=1)  # type: ignore
    ep_names = dataset.get_trajectory_names()
    ep_names.sort()
    print(f'{len(ep_names)} episodes in {conf.env}')

    for i in range(len(ep_names)):
        print(i, ep_names[i])
        episode = dataset.load_data(ep_names[i], skip_interval=0, include_metadata=False)

        imgs = []
        actions_vec = []
        rewards = []
        img_last = None

        for step in episode:
            # TODO: state
            imgs.append(step[0]['pov'])
            actions_vec.append(step[1]['vector'])
            rewards.append(step[2])
            img_last = step[3]['pov']

        while len(actions_vec) % ACTION_REPEAT != 0:
            # Pad end to make divisible by 4
            imgs.append(img_last)
            actions_vec.append(actions_vec[-1])
            rewards.append(0.0)

        # Group by action repeat
        imgs = imgs[::ACTION_REPEAT]
        rewards = np.array(rewards).reshape((-1, ACTION_REPEAT)).sum(-1)
        actions_vec = np.stack(actions_vec).reshape((-1, ACTION_REPEAT * 64))  # type: ignore
        distances = np.sum((actions_vec - action_centroids[:, None]) ** 2, -1)  # type: ignore
        actions = np.argmin(distances, axis=0)

        data = {
            'action': np.concatenate([[0], actions]),
            'image_t': np.stack(imgs + [img_last], -1),
            'reward': np.concatenate([[0.0], rewards]),
            'terminal': np.array([False] * len(rewards) + [True]),
            'reset': np.array([True] + [False] * len(rewards)),
        }

        if i == 0:
            print('Data sample: ', {k: v.shape for k, v in data.items()})

        fname = build_episode_name(0, i, i, int(data['reward'].sum()), len(data['reward']) - 1)
        mlflow_log_npz(data, fname, 'episodes', verbose=True)


if __name__ == "__main__":
    conf = args.parse_args()
    if conf.cluster:
        cluster_actions()
    save_episodes(conf)
