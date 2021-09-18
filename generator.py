import argparse
import collections
import datetime
import os
import pathlib
import time
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", ".*Box bound precision lowered by casting")

import gym
import mlflow
import numpy as np
import scipy.signal
import torch
import torch.distributions as D
from numba import njit

from envs import Atari, MiniGrid
from models import Dreamer, BehavioralCloning
from preprocessing import Preprocessor
from tools import *

WALL = 2


def create_env(env_id: str, max_steps: int, no_terminal: bool, seed: int):

    if env_id.startswith('MiniGrid-'):
        env = MiniGrid(env_id, seed=seed)

    elif env_id.startswith('Atari-'):
        env = Atari(env_id.split('-')[1].lower())

    elif env_id.startswith('AtariGray-'):
        env = Atari(env_id.split('-')[1].lower(), grayscale=True)

    elif env_id.startswith('MiniWorld-'):
        import gym_miniworld.wrappers as wrap
        env = env_raw = gym.make(env_id)
        env = wrap.DictWrapper(env)
        env = wrap.MapWrapper(env)
        # env = wrap.PixelMapWrapper(env)
        env = wrap.AgentPosWrapper(env)

    elif env_id.startswith('DmLab-'):
        from envs_dmlab import DmLab
        env = DmLab(env_id.split('-')[1].lower(), num_action_repeats=4)
        env = DictWrapper(env)

    elif env_id.startswith('MineRL'):
        from envs_minerl import MineRL
        env = MineRL(env_id, np.load('data/minerl_action_centroids_1.npy'), action_repeat=1)  # TODO

    else:
        env = gym.make(env_id)
        env = DictWrapper(env)

    env = ActionRewardResetWrapper(env, no_terminal)
    # TODO: TimeLimit(max_steps) wrapper, which wouldn't set the `terminal`
    env = CollectWrapper(env)
    return env


def main(env_id='MiniGrid-MazeS11N-v0',
         seed=0,
         policy='random',
         num_steps=int(1e6),
         env_max_steps=int(1e5),
         env_no_terminal=False,
         steps_per_npz=1500,
         model_reload_interval=60,
         model_conf=dict(),
         log_mlflow_metrics=True,
         eval_fraction=0.0,
         metrics_prefix='agent',
         ):

    # Mlflow

    if 'MLFLOW_RUN_ID' in os.environ:
        run = mlflow.active_run()
        if run is None:
            run = mlflow.start_run()
        print(f'Generator using existing mlflow run {run.info.run_id}')
    else:
        run = mlflow.start_run(run_name=f'{env_id}-s{seed}')
        print(f'Mlflow run {run.info.run_id} in experiment {run.info.experiment_id}')

    episodes_dir = 'episodes' if eval_fraction < 1.0 else 'episodes_eval'  # HACK
    artifact_dir = run.info.artifact_uri.replace('file://', '') + '/' + episodes_dir
    if artifact_dir.startswith('gs:/') or artifact_dir.startswith('s3:/'):
        artifact_dir = Pathy(artifact_dir)
    else:
        artifact_dir = Path(artifact_dir)

    # Env

    env = create_env(env_id, env_max_steps, env_no_terminal, seed)

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
                                  action_dim=env.action_size)
        policy = NetworkPolicy(model, preprocess)

    elif policy == 'random':
        policy = RandomPolicy(env.action_space)
    elif policy == 'minigrid_wander':
        policy = MinigridWanderPolicy()
    elif policy == 'maze_bouncing_ball':
        policy = MazeBouncingBallPolicy()
    elif policy == 'maze_dijkstra':
        step_size = env_raw.params.params['forward_step'].default / env_raw.room_size  # type: ignore
        turn_size = env_raw.params.params['turn_step'].default  # type: ignore
        policy = MazeDijkstraPolicy(step_size, turn_size)
    else:
        assert False, 'Unknown policy'

    # RUN

    steps, episodes = count_steps(artifact_dir, seed)
    datas = []
    visited_stats = []
    first_save = True
    first_episode = True
    last_model_load = 0
    model_step = 0

    while steps < num_steps:

        if model is not None:
            if time.time() - last_model_load > model_reload_interval:
                while True:
                    model_step = mlflow_load_checkpoint(policy.model, map_location='cpu')  # type: ignore
                    if model_step:
                        print(f'[GEN{seed:>2}]  Generator loaded model checkpoint {model_step}')
                        last_model_load = time.time()
                        break
                    else:
                        print(f'[GEN{seed:>2}]  Generator model checkpoint not found, waiting...')
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

        # Calculate visited (for MiniGrid/MiniWorld)

        # if 'agent_pos' in data:
        #     agent_pos = data['agent_pos']
        #     agent_pos = np.floor(agent_pos / 2)
        #     agent_pos_visited = len(np.unique(agent_pos, axis=0))
        #     visited_pct = agent_pos_visited / 25
        #     visited_stats.append(visited_pct)
        # else:
        #     visited_pct = np.nan

        # Log

        fps = epsteps / (time.time() - timer + 1e-6)
        if first_episode:
            print('Episode data sample: ', {k: v.shape for k, v in data.items()})
            first_episode = False

        print(f"[GEN{seed:>2}]  Episode recorded:"
              f"  steps: {epsteps}"
              f",  reward: {data['reward'].sum()}"
              f",  terminal: {data['terminal'].sum()}"
              f",  fps: {fps:.0f}"
              f",  total steps: {steps:.0f}"
              f",  episodes: {episodes}"
              )

        if log_mlflow_metrics:
            log_step = model_step if model else steps
            metrics = {f'{metrics_prefix}/{k}': np.mean(v) for k, v in metrics.items()}
            metrics.update({
                f'{metrics_prefix}/episode_length': epsteps,
                f'{metrics_prefix}/fps': fps,
                f'{metrics_prefix}/steps': steps,
                f'{metrics_prefix}/episodes': episodes,
                f'{metrics_prefix}/return': data['reward'].sum(),
            })  # type: ignore

            # Calculate return_discounted
            rewards_v = data['reward'].copy()
            if not data['terminal'][-1]:
                avg_value = rewards_v.mean() / (1.0 - model_conf.gamma)
                rewards_v[-1] += avg_value
            returns_discounted = discount(rewards_v, gamma=model_conf.gamma)
            metrics[f'{metrics_prefix}/return_discounted'] = returns_discounted.mean()

            # Calculate policy_value_terminal
            if data['terminal'][-1]:
                value_terminal = data['policy_value'][-2] - data['reward'][-1]  # This should be zero, because value[last] = reward[last]
                metrics[f'{metrics_prefix}/policy_value_terminal'] = value_terminal
            mlflow.log_metrics(metrics, step=log_step)

        # Save to npz

        datas.append(data)
        datas_episodes = len(datas)
        datas_steps = sum(len(d['reset']) - 1 for d in datas)
        datas_reward = sum(d['reward'].sum() for d in datas)

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

            if first_save:
                print('Saved data sample: ', {k: v.shape for k, v in data.items()})
                first_save = False

            fname = build_episode_name(seed, episodes - datas_episodes, episodes - 1, int(datas_reward), datas_steps)
            episodes_dir = 'episodes' if np.random.rand() > eval_fraction else 'episodes_eval'
            mlflow_log_npz(data, fname, episodes_dir, verbose=False)

    print(f'[GEN{seed:>2}]  Generator done.')


def discount(x: np.ndarray, gamma: float) -> np.ndarray:
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]  # type: ignore


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
    return (seed, episode, steps)


def count_steps(artifact_dir, seed):
    files = list(sorted(artifact_dir.glob('*.npz')))
    steps = 0
    episodes = 0
    for f in files:
        fseed, fepisode, fsteps = parse_episode_name(f.name)
        if fseed != seed:
            continue  # Belongs to another generator
        steps += fsteps
        episodes = max(episodes, fepisode + 1)

    print(f'Found existing {len(files)} files, {episodes} episodes, {steps} steps in {artifact_dir} (seed {seed})')
    return steps, episodes


class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def __call__(self, obs) -> Tuple[int, dict]:
        return self.action_space.sample(), {}


class NetworkPolicy:
    def __init__(self, model: Dreamer, preprocess: Preprocessor):
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


class MinigridWanderPolicy:
    def __call__(self, obs) -> Tuple[int, dict]:
        if obs['image'].shape == (7, 7):
            (ax, ay) = (3, 6)  # agent is here
            front = MiniGrid.GRID_VALUES[obs['image'][ax, ay - 1]]  # front is up
            left = MiniGrid.GRID_VALUES[obs['image'][ax - 1, ay]]
            right = MiniGrid.GRID_VALUES[obs['image'][ax + 1, ay]]
        elif 'map_centered' in obs:
            ax = ay = obs['map_centered'].shape[0] // 2  # agent is here
            front = MiniGrid.GRID_VALUES[obs['map_centered'][ax, ay - 1]]
            left = MiniGrid.GRID_VALUES[obs['map_centered'][ax - 1, ay]]
            right = MiniGrid.GRID_VALUES[obs['map_centered'][ax + 1, ay]]
        else:
            assert False, f'Unsupported observation {obs["image"].shape}'

        empty = [1, 8]  # Empty or goal

        # Door on left => turn with 50%
        if left[0] == 4 and np.random.rand() < 0.50:
            return 0, {}

        # Door on right => turn with 50%
        if right[0] == 4 and np.random.rand() < 0.50:
            return 1, {}

        # Empty left  => turn with 10%
        if left[0] in empty and np.random.rand() < 0.10:
            return 0, {}

        # Empty right => turn with 10%
        if right[0] in empty and np.random.rand() < 0.10:
            return 1, {}

        # Closed door => open
        if front[0] == 4 and front[2] == 1:
            return 5, {}

        # Empty or open door => forward
        if front[0] in empty or (front[0] == 4 and front[2] == 0):
            return 2, {}

        # If forward blocked...

        # If wall left and not right => turn right
        if left[0] == 2 and right[0] != 2:
            return 1, {}

        # If wall right and not left => turn left
        if right[0] == 2 and left[0] != 2:
            return 0, {}

        # Left-right 50%
        if np.random.rand() < 0.50:
            return 0, {}
        else:
            return 1, {}


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


class DictWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.observation_space = ...  # TODO

    def observation(self, obs_img):
        return {'image': obs_img}


class ActionRewardResetWrapper:

    def __init__(self, env, no_terminal):
        self._env = env
        self._no_terminal = no_terminal
        # Handle environments with one-hot or discrete action, but collect always as one-hot
        self.action_size = env.action_space.shape[0] if env.action_space.shape != () else env.action_space.n

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        if isinstance(action, int):
            action_onehot = np.zeros(self.action_size)
            action_onehot[action] = 1.0
        else:
            assert isinstance(action, np.ndarray) and action.shape == (self.action_size,), "Wrong one-hot action shape"
            action_onehot = action
        obs['action'] = action_onehot
        obs['reward'] = np.array(reward)
        obs['terminal'] = np.array(False if self._no_terminal else done)
        obs['reset'] = np.array(False)
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs['action'] = np.zeros(self.action_size)
        obs['reward'] = np.array(0.0)
        obs['terminal'] = np.array(False)
        obs['reset'] = np.array(True)
        return obs


class CollectWrapper:

    def __init__(self, env):
        self._env = env
        self._episode = []

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        self._episode.append(obs.copy())
        if done:
            episode = {k: np.array([t[k] for t in self._episode]) for k in self._episode[0]}
            info['episode'] = episode
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        self._episode = [obs.copy()]
        return obs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, required=True)
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--num_steps', type=int, default=1_000_000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--env_max_steps', type=int, default=int(1e5))
    args = parser.parse_args()
    main(**vars(args))
