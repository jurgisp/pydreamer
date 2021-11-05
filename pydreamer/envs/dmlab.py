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

import os
import gym
import numpy as np
from PIL import Image

import deepmind_lab  # type: ignore

# Default (action_dim=9)
# ACTION_SET = (  # IMPALA action set
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
ACTION_SET = {  # R2D2 action set
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

ALL_GAMES = frozenset([
    'rooms_collect_good_objects_train',  # rooms_collect_good_objects
    'rooms_collect_good_objects_test',  # rooms_collect_good_objects
    'rooms_exploit_deferred_effects_train',  # rooms_exploit_deferred_effects
    'rooms_exploit_deferred_effects_test',  # rooms_exploit_deferred_effects
    'rooms_select_nonmatching_object',
    'rooms_watermaze',
    'rooms_keys_doors_puzzle',
    'language_select_described_object',
    'language_select_located_object',
    'language_execute_random_task',
    'language_answer_quantitative_question',
    'lasertag_one_opponent_small',
    'lasertag_three_opponents_small',
    'lasertag_one_opponent_large',
    'lasertag_three_opponents_large',
    'natlab_fixed_large_map',
    'natlab_varying_map_regrowth',
    'natlab_varying_map_randomized',
    'skymaze_irreversible_path_hard',
    'skymaze_irreversible_path_varied',
    'psychlab_arbitrary_visuomotor_mapping',
    'psychlab_continuous_recognition',
    'psychlab_sequential_comparison',
    'psychlab_visual_search',
    'explore_object_locations_small',
    'explore_object_locations_large',
    'explore_obstructed_goals_small',
    'explore_obstructed_goals_large',
    'explore_goal_locations_small',
    'explore_goal_locations_large',
    'explore_object_rewards_few',
    'explore_object_rewards_many',
])


class DmLab(gym.Env):
    """DeepMind Lab wrapper."""

    def __init__(self, game, num_action_repeats, action_set=ACTION_SET):
        self.num_action_repeats = num_action_repeats
        self.env = deepmind_lab.Lab(
            level='contributed/dmlab30/' + game,
            observations=['RGB_INTERLEAVED'],
            config=dict(
                # fps='30',   # this produces 900 not 1800 steps in watermaze
                height='72',  # 72x96 to match RLU observations
                width='96',
                # dataset needed for psychlab_*, see https://github.com/deepmind/lab/tree/master/data/brady_konkle_oliva2008
                datasetPath=os.environ.get('DMLAB_DATASET_PATH', ''),
                maxAltCameraHeight='1',
                maxAltCameraWidth='1',
                hasAltCameras='false',
                allowHoldOutLevels='true',
                ),
        )
        self.action_set = action_set
        self.action_space = gym.spaces.Discrete(len(self.action_set))  # type: ignore
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)  # type: ignore

    def observation(self):
        img = self.env.observations()['RGB_INTERLEAVED']
        img = np.array(Image.fromarray(img).resize((64, 64), Image.NEAREST))
        return img

    def reset(self):
        self.env.reset()
        return self.observation()

    def step(self, action):
        raw_action = np.array(self.action_set[action], np.intc)
        reward = self.env.step(raw_action, num_steps=self.num_action_repeats)
        done = not self.env.is_running()
        if not done:
            observation = self.observation()
        else:
            # Do not have actual observation in done state, but need to return something
            observation = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)  # type: ignore
        return observation, reward, done, {}
