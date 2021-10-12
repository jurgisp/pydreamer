import argparse
import sys
import os
from pathlib import Path
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import torch
import torch.distributions as D
from numba import njit

sys.path.append(str(Path(__file__).parent))

from data import MlflowEpisodeRepository
from models import *
from preprocessing import Preprocessor
from tools import *
from envs import create_env

WALL = 2


def main(env_id='MiniGrid-MazeS11N-v0',
         save_uri=None,
         save_uri2=None,
         worker_id=0,
         policy='random',
         num_steps=int(1e6),
         env_no_terminal=False,
         env_time_limit=0,
         env_action_repeat=1,
         steps_per_npz=1750,
         model_reload_interval=120,
         model_conf=dict(),
         log_mlflow_metrics=True,
         split_fraction=0.0,
         metrics_prefix='agent',
         metrics_gamma=0.99,
         log_every=10,
         ):

    # Mlflow

    if 'MLFLOW_RUN_ID' in os.environ:
        run = mlflow.active_run()
        if run is None:
            run = mlflow.start_run(run_id=os.environ['MLFLOW_RUN_ID'])
    else:
        mlflow.start_run(run_name=f'{env_id}-{worker_id}')

    print(f'Generator {worker_id} started: env={env_id}, split_fraction={split_fraction}, metrics={metrics_prefix if log_mlflow_metrics else None}, save_uri={save_uri}')

    if not save_uri:
        save_uri = f'{mlflow.active_run().info.artifact_uri}/episodes'  # type: ignore
    if not save_uri2:
        assert split_fraction == 0.0, 'Specify two save destinations, if splitting'

    repository = MlflowEpisodeRepository(save_uri)
    repository2 = MlflowEpisodeRepository(save_uri2) if save_uri2 else repository
    nfiles, steps, episodes = repository.count_steps()
    print(f'Found existing {nfiles} files, {episodes} episodes, {steps} steps in {repository}')

    # Env

    env = create_env(env_id, env_no_terminal, env_time_limit, env_action_repeat)

    # Policy

    model = None
    if policy == 'network':
        conf = model_conf
        if conf.model == 'dreamer':
            model = Dreamer(conf)
        elif conf.model == 'bc':
            model = BehavioralCloning(conf)
        else:
            assert False
        preprocess = Preprocessor(image_categorical=conf.image_channels if conf.image_categorical else None,
                                  image_key=conf.image_key,
                                  map_categorical=conf.map_channels if conf.map_categorical else None,
                                  map_key=conf.map_key,
                                  action_dim=env.action_size,
                                  clip_rewards=conf.clip_rewards)
        policy = NetworkPolicy(model, preprocess)

    elif policy == 'random':
        policy = RandomPolicy(env.action_space)
    elif policy == 'minigrid_wander':
        from envs.minigrid import MinigridWanderPolicy
        policy = MinigridWanderPolicy()
    elif policy == 'maze_bouncing_ball':
        policy = MazeBouncingBallPolicy()
    elif policy == 'maze_dijkstra':
        step_size = env.params.params['forward_step'].default / env.room_size  # type: ignore
        turn_size = env.params.params['turn_step'].default  # type: ignore
        policy = MazeDijkstraPolicy(step_size, turn_size)
    else:
        assert False, 'Unknown policy'

    # RUN

    datas = []
    last_model_load = 0
    model_step = 0
    metrics_agg = defaultdict(list)

    while steps < num_steps:

        if model is not None:
            if time.time() - last_model_load > model_reload_interval:
                while True:
                    # takes ~10sec to load checkpoint
                    model_step = mlflow_load_checkpoint(policy.model, map_location='cpu')  # type: ignore
                    if model_step:
                        print(f'[GEN{worker_id:>2}]  Generator loaded model checkpoint {model_step}')
                        last_model_load = time.time()
                        break
                    else:
                        print(f'[GEN{worker_id:>2}]  Generator model checkpoint not found, waiting...')
                        time.sleep(10)

        # Unroll one episode

        epsteps = 0
        timer = time.time()
        obs = env.reset()
        done = False
        metrics = defaultdict(list)

        while not done:
            action, mets = policy(obs)
            obs, reward, done, info = env.step(action)
            steps += 1
            epsteps += 1
            for k, v in mets.items():
                metrics[k].append(v)

        episodes += 1
        data = info['episode']  # type: ignore
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

        print(f"[GEN{worker_id:>2}]  Episode recorded:"
              f"  steps: {epsteps}"
              f",  reward: {data['reward'].sum()}"
              f",  terminal: {data['terminal'].sum()}"
              f",  visited: {(data.get('map_seen', np.zeros(1))[-1] > 0).mean():.1%}"
              f",  total steps: {steps:.0f}"
              f",  episodes: {episodes}"
              f",  fps: {fps:.0f}"
              )

        if log_mlflow_metrics:
            metrics = {f'{metrics_prefix}/{k}': np.mean(v) for k, v in metrics.items()}
            metrics.update({
                f'{metrics_prefix}/episode_length': epsteps,
                f'{metrics_prefix}/fps': fps,
                f'{metrics_prefix}/steps': steps,
                f'{metrics_prefix}/env_steps': steps * env_action_repeat,
                f'{metrics_prefix}/episodes': episodes,
                f'{metrics_prefix}/return': data['reward'].sum(),
            })  # type: ignore

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

            # Aggregate every 10 episodes

            for k, v in metrics.items():
                if not np.isnan(v):
                    metrics_agg[k].append(v)

            if len(metrics_agg[f'{metrics_prefix}/return']) >= log_every:
                metrics_agg = {k: np.mean(v) for k, v in metrics_agg.items()}
                mlflow.log_metrics(metrics_agg, step=model_step if model else 0)
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

            # NHWC => HWCN for better compression

            data['image_t'] = data['image'].transpose(1, 2, 3, 0)
            del data['image']

            # Save to npz

            print_once('Collected data sample: ', {k: v.shape for k, v in data.items()})
            if np.random.rand() > split_fraction:
                repository.save_data(data, episodes - datas_episodes, episodes - 1)
            else:
                repository2.save_data(data, episodes - datas_episodes, episodes - 1)

    print(f'[GEN{worker_id:>2}]  Generator done.')


class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def __call__(self, obs) -> Tuple[int, dict]:
        return self.action_space.sample(), {}


class NetworkPolicy:
    def __init__(self, model: TrainableModel, preprocess: Preprocessor):
        self.model = model
        self.preprocess = preprocess
        self._state = model.init_state(1)

    def __call__(self, obs) -> Tuple[int, dict]:
        batch = self.preprocess.apply(obs, expandTB=True)

        image = torch.from_numpy(batch['image'])
        reward = torch.from_numpy(batch['reward'])
        action = torch.from_numpy(batch['action'])
        reset = torch.from_numpy(batch['reset'])

        with torch.no_grad():
            action_logits, value, new_state = self.model.forward(image, reward, action, reset, self._state)
            action_logits = action_logits[0, 0]  # (N=1,B=1,A) => (A)
            value = value[0, 0]
            action_distr = D.OneHotCategorical(logits=action_logits)
            self._state = new_state

        action = action_distr.sample()
        action = action.argmax(-1)  # one-hot => int

        metrics = dict(policy_value=value.item(),
                       action_prob=action_distr.probs[action].item(),
                       policy_entropy=action_distr.entropy().item())

        return action.item(), metrics


class MazeBouncingBallPolicy:
    # Policy:
    #   1) Forward until you hit a wall
    #   2) Turn in random 360 direction
    #   3) Go to 1)

    def __init__(self):
        self.pos = None
        self.turns_remaining = 0

    def __call__(self, obs) -> Tuple[int, dict]:
        assert 'agent_pos' in obs, f'Need agent position'
        pos = obs['agent_pos']
        action = -1

        # print(f'{self.pos} => {pos} ({obs["agent_dir"]})')

        if self.turns_remaining == 0:
            if self.pos is None or not np.all(self.pos == pos):
                # Going forward
                action = 2
                self.pos = pos
            else:
                # Hit the wall - start turning
                if np.random.randint(2) == 0:
                    # self.turns_remaining = -np.random.randint(2, 5)  # Left
                    self.turns_remaining = -1  # TODO
                else:
                    # self.turns_remaining = np.random.randint(2, 5)  # Right
                    self.turns_remaining = 1  # TODO
                self.pos = None

        if self.turns_remaining > 0:
            # Turning right
            action = 1
            self.turns_remaining -= 1

        elif self.turns_remaining < 0:
            # Turning left
            action = 0
            self.turns_remaining += 1

        assert action >= 0
        return action, {}


class MazeDijkstraPolicy:
    # Policy:
    #   1) Pick a random spot on a map
    #   2) Go there using shortest path
    #   3) Occasionally perform a random action

    def __init__(self, step_size, turn_size, epsilon=0.10):
        self.step_size = step_size
        self.turn_size = turn_size
        self.epsilon = epsilon
        self._goal = None
        self._expected_pos = None

    def __call__(self, obs) -> Tuple[int, dict]:
        assert 'agent_pos' in obs, 'Need agent position'
        assert 'map_agent' in obs, 'Need map'

        x, y = obs['agent_pos']
        dx, dy = obs['agent_dir']
        d = np.arctan2(dy, dx) / np.pi * 180  # type: ignore
        map = obs['map_agent']
        # assert map[int(x), int(y)] >= 3, 'Agent should be here'

        if obs['reset']:
            self._goal = None  # new episode
            self._expected_pos = None
        if self._goal is None:
            self._goal = self._generate_goal(map)

        if self._expected_pos is not None:
            if not np.isclose(self._expected_pos[:2], [x, y], 1e-3).all():
                print('WARN: unexpected position - stuck? Generating new goal...')
                self._goal = self._generate_goal(map)

        while True:
            t = time.time()
            actions, path, nvis = find_shortest(map, (x, y, d), self._goal, self.step_size, self.turn_size)
            # print(f'Pos: {tuple(np.round([x,y,d], 2))}'
            #       f', Goal: {self._goal}'
            #       f', Len: {len(actions)}'
            #       f', Actions: {actions[:1]}'
            #       # f', Path: {path[:1]}'
            #       f', Visited: {nvis}'
            #       f', Time: {int((time.time()-t)*1000)}'
            #       )
            if len(actions) > 0:
                if np.random.rand() < self.epsilon:
                    self._expected_pos = None
                    return np.random.randint(3), {}  # random action
                else:
                    self._expected_pos = path[0]
                    return actions[0], {}  # best action
            else:
                self._goal = self._generate_goal(map)

    @staticmethod
    def _generate_goal(map):
        while True:
            x = np.random.randint(map.shape[0])
            y = np.random.randint(map.shape[1])
            if map[x, y] != WALL:
                return x, y


@njit
def find_shortest(map, start, goal, step_size=1.0, turn_size=45.0):
    KPREC = 5
    RADIUS = 0.2
    x, y, d = start
    gx, gy = goal

    # Well ok, this is BFS not Dijkstra, technically speaking

    que = []
    que_ix = 0
    visited = {}
    parent = {}
    parent_action = {}

    p = (x, y, d)
    key = (round(x * KPREC) / KPREC, round(y * KPREC) / KPREC, round(d * KPREC) / KPREC)
    que.append(p)
    visited[key] = True
    goal_state = None

    while que_ix < len(que):
        p = que[que_ix]
        que_ix += 1
        x, y, d = p
        if int(x) == int(gx) and int(y) == int(gy):
            goal_state = p
            break
        for action in range(3):
            x1, y1, d1 = x, y, d
            if action == 0:  # turn left
                d1 = d - turn_size
                if d1 < -180.0:
                    d1 += 360.0
            if action == 1:  # turn right
                d1 = d + turn_size
                if d1 > 180.0:
                    d1 -= 360.0
            if action == 2:  # forward
                x1 = x + step_size * np.cos(d / 180 * np.pi)
                y1 = y + step_size * np.sin(d / 180 * np.pi)
                # Check wall collision at 4 corners
                for x2, y2 in [(x1 - RADIUS, y1 - RADIUS), (x1 + RADIUS, y1 - RADIUS), (x1 - RADIUS, y1 + RADIUS), (x1 + RADIUS, y1 + RADIUS)]:
                    if x2 < 0 or y2 < 0 or x2 >= map.shape[0] or y2 >= map.shape[1] or map[int(x2), int(y2)] == WALL:
                        x1, y1 = x, y  # wall
                        break
            p1 = (x1, y1, d1)
            key = (round(x1 * KPREC) / KPREC, round(y1 * KPREC) / KPREC, round(d1 * KPREC) / KPREC)
            if key not in visited:
                que.append(p1)
                parent[p1] = p
                parent_action[p1] = action
                visited[key] = True
                assert len(visited) < 100000, 'Runaway Dijkstra'

    path = []
    actions = []
    if goal_state is not None:
        p = goal_state
        while p in parent_action:
            path.append(p)
            actions.append(parent_action[p])
            p = parent[p]
        path.reverse()
        actions.reverse()
    else:
        print('WARN: no path found')

    return actions, path, len(visited)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, required=True)
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--save_uri', type=str, default='')
    parser.add_argument('--num_steps', type=int, default=1_000_000)
    parser.add_argument('--worker_id', type=int, default=0)
    parser.add_argument('--env_time_limit', type=int, default=0)
    parser.add_argument('--env_action_repeat', type=int, default=1)
    args = parser.parse_args()
    main(**vars(args))
