import argparse
import numpy as np
import datetime
import time
import pathlib
import gym
from envs import MiniGrid


def main(output_dir,
         env_name,
         num_steps=1_000_000,
         policy='minigrid_wander'
         ):

    output_dir = pathlib.Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = MiniGrid(env_name, max_steps=500)
    env = CollectWrapper(env)

    if policy == 'random':
        policy = RandomPolicy(env.action_space)
    elif policy == 'minigrid_wander':
        policy = MinigridWanderPolicy()
    else:
        assert False, 'Unknown policy'

    steps, episodes = 0, 0
    while steps < num_steps:

        # Unroll one episode

        obs, done = env.reset(), False
        epsteps, timer = 0, time.time()
        while not done:
            action = policy(obs)
            obs, reward, done, info = env.step(action)
            steps += 1
            epsteps += 1
        data = info['episode']

        # Save to npz

        fname = output_dir / f'ep{episodes:06}-{epsteps:04}.npz'
        with fname.open('wb') as f:
            np.savez_compressed(f, **data)

        episodes += 1
        fps = epsteps / (time.time() - timer + 1e-6)
        if episodes == 1:
            print('Data sample: ', {k: v.shape for k, v in data.items()})

        print(f"[{steps:08}/{num_steps:08}] "
              f"Episode data: {data['image'].shape} written to {fname}"
              f",  explored%: {(data['map_vis'][-1] < 1000).mean():.3f}"
              f",  fps: {fps:.0f}"
              )


class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def __call__(self, obs):
        return self.action_space.sample()


class MinigridWanderPolicy:
    def __call__(self, obs):
        front = MiniGrid.GRID_VALUES[obs['image'][3, 5]]
        left = MiniGrid.GRID_VALUES[obs['image'][2, 6]]
        right = MiniGrid.GRID_VALUES[obs['image'][4, 6]]

        # Door on left => turn with 50%
        if left[0] == 4 and np.random.rand() < 0.50:
            return 0

        # Door on right => turn with 50%
        if right[0] == 4 and np.random.rand() < 0.50:
            return 1

        # Empty left  => turn with 10%
        if left[0] == 1 and np.random.rand() < 0.10:
            return 0

        # Empty right => turn with 10%
        if right[0] == 1 and np.random.rand() < 0.10:
            return 1

        # Closed door => open
        if front[0] == 4 and front[2] == 1:
            return 5

        # Empty or open door => forward
        if front[0] == 1 or (front[0] == 4 and front[2] == 0):
            return 2

        # If forward blocked...

        # If wall left and not right => turn right
        if left[0] == 2 and right[0] != 2:
            return 1

        # If wall right and not left => turn left
        if right[0] == 2 and left[0] != 2:
            return 0

        # Left-right 50%
        if np.random.rand() < 0.50:
            return 0
        else:
            return 1


class CollectWrapper:

    def __init__(self, env):
        self._env = env
        self._episode = None
        # Handle environments with one-hot or discrete action, but collect always as one-hot
        self._action_size = env.action_space.shape[0] if env.action_space.shape != () else env.action_space.n

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        transition = obs.copy()

        if isinstance(action, int):
            action_onehot = np.zeros(self._action_size)
            action_onehot[action] = 1.0
        else:
            assert isinstance(action, np.ndarray) and action.shape == (self._action_size,), "Wrong one-hot action shape"
            action_onehot = action
        transition['action'] = action_onehot
        transition['reward'] = reward
        transition['discount'] = np.array(1 - float(done))

        self._episode.append(transition)
        if done:
            episode = {k: np.array([t[k] for t in self._episode]) for k in self._episode[0]}
            info['episode'] = episode
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        transition = obs.copy()
        transition['action'] = np.zeros(self._action_size)
        transition['reward'] = 0.0
        transition['discount'] = 1.0
        self._episode = [transition]
        return obs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env')
    args = parser.parse_args()

    env = args.env
    main(output_dir=f"data/{env}/{datetime.datetime.now().strftime('%Y%m%d_%H%M')}",
         env_name=env)
