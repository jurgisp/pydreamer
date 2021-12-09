# Copyright danijar, jurgisp

import gym
import gym.spaces
import minerl
import numpy as np


def make_action(pitch=0, yaw=0, **kwargs):
    action = dict(
        camera=[pitch, yaw],
        forward=0, back=0, left=0, right=0,
        attack=0, sprint=0, jump=0, sneak=0)
    action.update(kwargs)
    return action


BASIC_ACTIONS = (
    make_action(),
    make_action(pitch=-10),
    make_action(pitch=10),
    make_action(yaw=-30),
    make_action(yaw=30),
    make_action(attack=1),
    make_action(forward=1),
    make_action(back=1),
    make_action(left=1),
    make_action(right=1),
    make_action(sprint=1),
    make_action(jump=1, forward=1),
)


class MineRL(gym.Env):
    """DeepMind Lab wrapper."""

    def __init__(self, env_id, action_set=BASIC_ACTIONS, action_repeat=1):
        self.env = gym.make(env_id)
        self.action_set = self.extend_with_enum_actions(action_set)
        self.action_repeat = action_repeat

        self._inv_keys = [k for k in self.env.observation_space['inventory'].spaces]  # type: ignore
        self._equip_enum = self.env.observation_space['equipped_items']['mainhand']['type'].values.tolist()  # type: ignore

        self.action_space = gym.spaces.Discrete(len(self.action_set))
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(shape=(64, 64, 3), low=0, high=255, dtype=np.uint8),
            'inventory': gym.spaces.Box(shape=(len(self._inv_keys),), low=0, high=np.inf, dtype=np.float32),
            'equipped': gym.spaces.Box(shape=(len(self._equip_enum),), low=0, high=1, dtype=np.float32),
        })

    def reset(self):
        obs = self.env.reset()
        return self.observation(obs)

    def step(self, action):
        act = self.action_set[action]
        reward = 0
        done = False
        for _ in range(self.action_repeat):
            obs, rew, done, info = self.env.step(act)
            reward += rew
            if done:
                break
        return self.observation(obs), reward, done, info  # type: ignore

    def observation(self, obs):
        inventory = np.array([obs['inventory'][k] for k in self._inv_keys])
        inventory = np.log(1 + np.array(inventory))
        index = self._equip_enum.index(obs['equipped_items']['mainhand']['type'])
        equipped = np.zeros(len(self._equip_enum), np.float32)
        equipped[index] = 1.0
        return {
            'image': obs['pov'],
            'inventory': inventory,
            'equipped': equipped,
        }

    def extend_with_enum_actions(self, action_set):
        action_set = [action.copy() for action in action_set]
        assert all(x in (0, [0, 0]) for x in action_set[0].values()), (
            f'first action should be noop but is {action_set[0]}')
        # Parse enum spaces from environment.
        enums = {}
        defaults = {}
        for key, space in self.env.action_space.spaces.items():  # type: ignore
            if type(space).__name__ == 'Enum':
                enums[key] = list(space.values)
                defaults[key] = space.default
        # Include enum keys in existing actions.
        for action in action_set:
            for key, values in enums.items():
                action[key] = values.index(defaults[key])
        # Create new actions for the enums.
        for key, values in sorted(enums.items()):
            for index, value in enumerate(values):
                if value == defaults[key]:
                    continue
                action = action_set[0].copy()
                action[key] = index
                action_set.append(action)
        # Swap enum integer values to their strings.
        for action in action_set:
            for key, enum in enums.items():
                action[key] = enum[action[key]]
        return tuple(action_set)
