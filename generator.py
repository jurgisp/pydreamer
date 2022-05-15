import argparse
import logging
import logging.config
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from itertools import chain
from logging import critical, debug, error, info, warning
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import torch
import torch.distributions as D

from pydreamer.data import MlflowEpisodeRepository
from pydreamer.envs import create_env
from pydreamer.models import *
from pydreamer.models.functions import map_structure
from pydreamer.preprocessing import Preprocessor
from pydreamer.tools import *


def main(env_id='MiniGrid-MazeS11N-v0',
         save_uri=None,
         save_uri2=None,
         worker_id=0,
         policy_main='random',
         policy_prefill='random',
         num_steps=int(1e6),
         num_steps_prefill=0,
         env_no_terminal=False,
         env_time_limit=0,
         env_action_repeat=1,
         limit_step_ratio=0.0,
         steps_per_npz=1000,
         model_reload_interval=120,
         model_conf=dict(),
         log_mlflow_metrics=True,
         split_fraction=0.0,
         metrics_prefix='agent',
         metrics_gamma=0.99,
         log_every=10,
         ):

    configure_logging(prefix=f'[GEN {worker_id}]', info_color=LogColorFormatter.GREEN)
    mlrun = mlflow_init()
    info(f'Generator {worker_id} started:'
         f' env={env_id}'
         f', n_steps={num_steps:,}'
         f', n_prefill={num_steps_prefill:,}'
         f', split_fraction={split_fraction}'
         f', metrics={metrics_prefix if log_mlflow_metrics else None}'
         f', save_uri={save_uri}')

    if not save_uri:
        save_uri = f'{mlrun.info.artifact_uri}/episodes/{worker_id}'
    if not save_uri2:
        assert split_fraction == 0.0, 'Specify two save destinations, if splitting'

    repository = MlflowEpisodeRepository(save_uri)
    repository2 = MlflowEpisodeRepository(save_uri2) if save_uri2 else repository
    nfiles, steps_saved, episodes = repository.count_steps()
    info(f'Found existing {nfiles} files, {episodes} episodes, {steps_saved} steps in {repository}')

    # Env

    env = create_env(env_id, env_no_terminal, env_time_limit, env_action_repeat, worker_id)

    # Policy

    if num_steps_prefill:
        # Start with prefill policy
        info(f'Prefill policy: {policy_prefill}')
        policy = create_policy(policy_prefill, env, model_conf)
        is_prefill_policy = True
    else:
        info(f'Policy: {policy_main}')
        policy = create_policy(policy_main, env, model_conf)
        is_prefill_policy = False

    # RUN

    datas = []
    last_model_load = 0
    model_step = 0
    metrics_agg = defaultdict(list)
    all_returns = []
    steps = 0

    while steps_saved < num_steps:

        # Switch policy prefill => main

        if is_prefill_policy and steps_saved >= num_steps_prefill:
            info(f'Switching to main policy: {policy_main}')
            policy = create_policy(policy_main, env, model_conf)
            is_prefill_policy = False

        # Load network

        if isinstance(policy, NetworkPolicy):
            if time.time() - last_model_load > model_reload_interval:
                while True:
                    # takes ~10sec to load checkpoint
                    model_step = mlflow_load_checkpoint(policy.model, map_location='cpu')  # type: ignore
                    if model_step:
                        info(f'Generator loaded model checkpoint {model_step}')
                        last_model_load = time.time()
                        break
                    else:
                        debug('Generator model checkpoint not found, waiting...')
                        time.sleep(10)

            if limit_step_ratio and steps_saved >= model_step * limit_step_ratio:
                # Rate limiting - keep looping until new model checkpoint is loaded
                time.sleep(1)
                continue

        # Unroll one episode

        epsteps = 0
        timer = time.time()
        obs = env.reset()
        done = False
        metrics = defaultdict(list)

        while not done:
            action, mets = policy(obs)
            obs, reward, done, inf = env.step(action)
            steps += 1
            epsteps += 1
            for k, v in mets.items():
                metrics[k].append(v)

        episodes += 1
        data = inf['episode']  # type: ignore
        if 'policy_value' in metrics:
            data['policy_value'] = np.array(metrics['policy_value'] + [np.nan])     # last terminal value is null
            data['policy_entropy'] = np.array(metrics['policy_entropy'] + [np.nan])  # last policy is null
            data['action_prob'] = np.array([np.nan] + metrics['action_prob'])       # first action is null
        else:
            # Need to fill with placeholders, so all batches have the same keys
            data['policy_value'] = np.full(data['reward'].shape, np.nan)
            data['policy_entropy'] = np.full(data['reward'].shape, np.nan)
            data['action_prob'] = np.full(data['reward'].shape, np.nan)

        # Log

        fps = epsteps / (time.time() - timer + 1e-6)
        print_once('Episode data sample: ', {k: v.shape for k, v in data.items()})

        info(f"Episode recorded:"
             f"  steps: {epsteps}"
             f",  reward: {data['reward'].sum():.1f}"
             f",  terminal: {data['terminal'].sum():.0f}"
            #  f",  visited: {(data.get('map_seen', np.zeros(1))[-1] > 0).mean():.1%}"
             f",  total steps: {steps:.0f}"
             f",  episodes: {episodes}"
             f",  saved steps (train): {steps_saved:.0f}"
             f",  fps: {fps:.0f}"
             )

        if log_mlflow_metrics:
            metrics = {f'{metrics_prefix}/{k}': np.array(v).mean() for k, v in metrics.items()}
            all_returns.append(data['reward'].sum())
            metrics.update({
                f'{metrics_prefix}/episode_length': epsteps,
                f'{metrics_prefix}/fps': fps,
                f'{metrics_prefix}/steps': steps,  # All steps since previous restart
                f'{metrics_prefix}/steps_saved': steps_saved,  # Steps saved in the training repo
                f'{metrics_prefix}/env_steps': steps * env_action_repeat,
                f'{metrics_prefix}/episodes': episodes,
                f'{metrics_prefix}/return': data['reward'].sum(),
                f'{metrics_prefix}/return_cum': np.array(all_returns[-100:]).mean(),
            })

            # Calculate return_discounted

            rewards_v = data['reward'].copy()
            if not data['terminal'][-1]:
                avg_value = rewards_v.mean() / (1.0 - metrics_gamma)
                rewards_v[-1] += avg_value
            returns_discounted = discount(rewards_v, gamma=metrics_gamma)
            metrics[f'{metrics_prefix}/return_discounted'] = returns_discounted.mean()

            # Calculate policy_value_terminal

            if data['terminal'][-1]:
                value_terminal = data['policy_value'][-2] - data['reward'][-1]  # This should be zero, because value[last] = reward[last]
                metrics[f'{metrics_prefix}/policy_value_terminal'] = value_terminal

            # Goal visibility metrics for Scavenger

            if 'goals_visage' in data:
                goals_seen = data['goals_visage'] < 1e5
                metrics[f'{metrics_prefix}/goals_seen_avg'] = goals_seen.sum(axis=-1).mean()
                metrics[f'{metrics_prefix}/goals_seen_last'] = goals_seen[-1].sum()
                metrics[f'{metrics_prefix}/goals_seenage'] = (data['goals_visage'] * goals_seen).sum() / goals_seen.sum()

            # Aggregate every 10 episodes

            for k, v in metrics.items():
                if not np.isnan(v):
                    metrics_agg[k].append(v)

            if len(metrics_agg[f'{metrics_prefix}/return']) >= log_every:
                metrics_agg_max = {k: np.array(v).max() for k, v in metrics_agg.items()}
                metrics_agg = {k: np.array(v).mean() for k, v in metrics_agg.items()}
                metrics_agg[f'{metrics_prefix}/return_max'] = metrics_agg_max[f'{metrics_prefix}/return']
                metrics_agg['_timestamp'] = datetime.now().timestamp()
                mlflow_log_metrics(metrics_agg, step=model_step)
                metrics_agg = defaultdict(list)

        # Save to npz

        datas.append(data)
        datas_episodes = len(datas)
        datas_steps = sum(len(d['reset']) - 1 for d in datas)

        if datas_steps >= steps_per_npz:

            # Concatenate episodes

            data = {}
            for key in datas[0]:
                data[key] = np.concatenate([b[key] for b in datas], axis=0)
            datas = []
            print_once('Collected data sample: ', {k: v.shape for k, v in data.items()})

            # ... or chunk

            # if steps_per_npz=1000, then chunk size will be [1000,1999]
            if datas_steps >= 2 * steps_per_npz:
                chunks = chunk_episode_data(data, steps_per_npz)
            else:
                chunks = [data]

            # Save to npz

            repo = repository if (np.random.rand() > split_fraction) else repository2
            for i, data in enumerate(chunks):
                if 'image' in data and len(data['image'].shape) == 4:
                    # THWC => HWCT for better compression
                    data['image_t'] = data['image'].transpose(1, 2, 3, 0)
                    del data['image']
                else:
                    # Categorical image, leave it alone
                    pass
                repo.save_data(data, episodes - datas_episodes, episodes - 1, i)

            if repo == repository:
                # Only count steps in the training repo, so that prefill and limit_step_ratio works correctly
                steps_saved += datas_steps

    info('Generator done.')


def create_policy(policy_type: str, env, model_conf):
    if policy_type == 'network':
        conf = model_conf
        if conf.model == 'dreamer':
            model = Dreamer(conf)
        else:
            assert False, conf.model
        preprocess = Preprocessor(image_categorical=conf.image_channels if conf.image_categorical else None,
                                  image_key=conf.image_key,
                                  map_categorical=conf.map_channels if conf.map_categorical else None,
                                  map_key=conf.map_key,
                                  action_dim=env.action_size,  # type: ignore
                                  clip_rewards=conf.clip_rewards)
        return NetworkPolicy(model, preprocess)

    if policy_type == 'random':
        return RandomPolicy(env.action_space)

    if policy_type == 'minigrid_wander':
        from pydreamer.envs.minigrid import MinigridWanderPolicy
        return MinigridWanderPolicy()

    if policy_type == 'maze_bouncing_ball':
        from pydreamer.envs.miniworld import MazeBouncingBallPolicy
        return MazeBouncingBallPolicy()

    if policy_type == 'maze_dijkstra':
        from pydreamer.envs.miniworld import MazeDijkstraPolicy
        step_size = env.params.params['forward_step'].default / env.room_size  # type: ignore
        turn_size = env.params.params['turn_step'].default  # type: ignore
        return MazeDijkstraPolicy(step_size, turn_size)

    if policy_type == 'goal_dijkstra':
        from pydreamer.envs.miniworld import MazeDijkstraPolicy
        step_size = env.params.params['forward_step'].default / env.room_size  # type: ignore
        turn_size = env.params.params['turn_step'].default  # type: ignore
        return MazeDijkstraPolicy(step_size, turn_size, goal_strategy='goal_direction', random_prob=0)

    raise ValueError(policy_type)


class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def __call__(self, obs) -> Tuple[int, dict]:
        return self.action_space.sample(), {}


class NetworkPolicy:
    def __init__(self, model: Dreamer, preprocess: Preprocessor):
        self.model = model
        self.preprocess = preprocess
        self.state = model.init_state(1)

    def __call__(self, obs) -> Tuple[np.ndarray, dict]:
        batch = self.preprocess.apply(obs, expandTB=True)
        obs_model: Dict[str, Tensor] = map_structure(batch, torch.from_numpy)  # type: ignore

        with torch.no_grad():
            action_distr, new_state, metrics = self.model.inference(obs_model, self.state)
            action = action_distr.sample()
            self.state = new_state

        metrics = {k: v.item() for k, v in metrics.items()}
        metrics.update(action_prob=action_distr.log_prob(action).exp().mean().item(),
                       policy_entropy=action_distr.entropy().mean().item())

        action = action.squeeze()  # (1,1,A) => A
        return action.numpy(), metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, required=True)
    parser.add_argument('--policy_main', type=str, required=True)
    parser.add_argument('--save_uri', type=str, default='')
    parser.add_argument('--num_steps', type=int, default=1_000_000)
    parser.add_argument('--worker_id', type=int, default=0)
    parser.add_argument('--env_time_limit', type=int, default=0)
    parser.add_argument('--env_action_repeat', type=int, default=1)
    parser.add_argument('--steps_per_npz', type=int, default=1000)
    args = parser.parse_args()
    main(**vars(args))
