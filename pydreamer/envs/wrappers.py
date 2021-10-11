import gym
import numpy as np


class DictWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.observation_space = ...  # TODO

    def observation(self, obs_img):
        return {'image': obs_img}


class TimeLimitWrapper(gym.Wrapper):

    def __init__(self, env, time_limit):
        super().__init__(env)
        self._time_limit = time_limit

    def step(self, action):
        obs, reward, done, info = self.env.step(action)  # type: ignore
        self._step += 1
        if self._step >= self._time_limit:
            done = True
            info['time_limit'] = True
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self.env.reset()  # type: ignore


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
        obs['terminal'] = np.array(False if self._no_terminal or info.get('time_limit') else done)
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
