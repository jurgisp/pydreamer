import gym
import minerl
import numpy as np


class MineRL(gym.Env):
    """DeepMind Lab wrapper."""

    def __init__(self, env_id, action_set, action_repeat):
        self.env = gym.make(env_id)
        self.action_set = action_set
        self.action_repeat = action_repeat
        self.action_space = gym.spaces.Discrete(len(self.action_set))  # type: ignore
        self.observation_space = gym.spaces.Dict({  # type: ignore
            'image': gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),  # type: ignore
            'vecobs': gym.spaces.Box(low=-1, high=1, shape=(64,), dtype=np.float32)  # type: ignore
        })

    def reset(self):
        obs = self.env.reset()
        return self.observation(obs)

    def step(self, action):
        action_vec = self.action_set[action]
        actions = action_vec.reshape((-1, 64))  # If action is [256] means it's 4x action repeat of [64] actions
        reward = 0
        done = False
        for _ in range(self.action_repeat):
            for act in actions:
                obs, rew, done, info = self.env.step({'vector': act})
                reward += rew
                if done:
                    break
            if done:
                break
        return self.observation(obs), reward, done, info  # type: ignore

    def observation(self, obs):
        return {
            'image': obs['pov'],
            'vecobs': obs['vector']
        }
