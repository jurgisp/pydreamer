# Copyright 2019 The SEED Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DeepMind Lab Gym wrapper."""

import gym
import numpy as np

import deepmind_lab  # type: ignore


# Default (action_dim=9)
# ACTION_SET = (
#     (0, 0, 0, 1, 0, 0, 0),    # Forward
#     (0, 0, 0, -1, 0, 0, 0),   # Backward
#     (0, 0, -1, 0, 0, 0, 0),   # Strafe Left
#     (0, 0, 1, 0, 0, 0, 0),    # Strafe Right
#     (-20, 0, 0, 0, 0, 0, 0),  # Look Left
#     (20, 0, 0, 0, 0, 0, 0),   # Look Right
#     (-20, 0, 0, 1, 0, 0, 0),  # Look Left + Forward
#     (20, 0, 0, 1, 0, 0, 0),   # Look Right + Forward
#     (0, 0, 0, 0, 1, 0, 0),    # Fire.
# )

# RLU (action:dim=15)
ACTION_SET = {
    0: (0, 0, 0, 1, 0, 0, 0),     # Forward
    1: (0, 0, 0, -1, 0, 0, 0),    # Backward
    2: (0, 0, -1, 0, 0, 0, 0),    # Strafe Left
    3: (0, 0, 1, 0, 0, 0, 0),     # Strafe Right
    4: (-10, 0, 0, 0, 0, 0, 0),   # Left (10 deg)
    5: (10, 0, 0, 0, 0, 0, 0),    # Right (10 deg)
    6: (-60, 0, 0, 0, 0, 0, 0),   # Left (60 deg)
    7: (60, 0, 0, 0, 0, 0, 0),    # Right (60 deg)
    8: (0, 10, 0, 0, 0, 0, 0),    # Up (10 deg)
    9: (0, -10, 0, 0, 0, 0, 0),   # Down (10 deg)
    10: (-10, 0, 0, 1, 0, 0, 0),  # Left (10 deg) + Forward
    11: (10, 0, 0, 1, 0, 0, 0),   # Right (10 deg) + Forward
    12: (-60, 0, 0, 1, 0, 0, 0),  # Left (60 deg) + Forward
    13: (60, 0, 0, 1, 0, 0, 0),   # Right (60 deg) + Forward
    14: (0, 0, 0, 0, 1, 0, 0),    # Fire
}


class DmLab(gym.Env):
    """DeepMind Lab wrapper."""

    def __init__(self, game, num_action_repeats, action_set=ACTION_SET):
        self._num_action_repeats = num_action_repeats
        self._env = deepmind_lab.Lab(
            level='contributed/dmlab30/' + game,
            observations=['RGB_INTERLEAVED'],
            config=dict(
                # fps='30',   # this produces 900 not 1800 steps in watermaze
                height='72',  # 72x96 to match RLU observations
                width='96',
                datasetPath='',  # dataset needed for psychlab_*, see https://github.com/deepmind/lab/tree/master/data/brady_konkle_oliva2008
                maxAltCameraHeight='1',
                maxAltCameraWidth='1',
                hasAltCameras='false'),
        )
        self._action_set = action_set
        self.action_space = gym.spaces.Discrete(len(self._action_set))  # type: ignore
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)  # type: ignore

    def _observation(self):
        img = self._env.observations()['RGB_INTERLEAVED']
        H, W = img.shape[:2]
        img = img[-64:, (W - 64) // 2:-(W - 64) // 2]  # center-bottom crop
        return img

    def reset(self):
        self._env.reset()
        return self._observation()

    def step(self, action):
        raw_action = np.array(self._action_set[action], np.intc)
        reward = self._env.step(raw_action, num_steps=self._num_action_repeats)
        done = not self._env.is_running()
        if not done:
            observation = self._observation()
        else:
            # Do not have actual observation in done state, but need to return something
            observation = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)  # type: ignore
        return observation, reward, done, {}
