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

import hashlib
import os
import gym
import numpy as np

import deepmind_lab


DEFAULT_ACTION_SET = (
    (0, 0, 0, 1, 0, 0, 0),    # Forward
    (0, 0, 0, -1, 0, 0, 0),   # Backward
    (0, 0, -1, 0, 0, 0, 0),   # Strafe Left
    (0, 0, 1, 0, 0, 0, 0),    # Strafe Right
    (-20, 0, 0, 0, 0, 0, 0),  # Look Left
    (20, 0, 0, 0, 0, 0, 0),   # Look Right
    (-20, 0, 0, 1, 0, 0, 0),  # Look Left + Forward
    (20, 0, 0, 1, 0, 0, 0),   # Look Right + Forward
    (0, 0, 0, 0, 1, 0, 0),    # Fire.
)

ALL_GAMES = frozenset([
    'rooms_collect_good_objects_train',
    'rooms_collect_good_objects_test',
    'rooms_exploit_deferred_effects_train',
    'rooms_exploit_deferred_effects_test',
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

    def __init__(self, game, num_action_repeats, seed, is_test, config,
                 action_set=DEFAULT_ACTION_SET, level_cache_dir=None):
        if is_test:
            config['allowHoldOutLevels'] = 'true'
            # Mixer seed for evalution, see
            # https://github.com/deepmind/lab/blob/master/docs/users/python_api.md
            config['mixerSeed'] = 0x600D5EED

        assert game in ALL_GAMES
        game = 'contributed/dmlab30/' + game

        # Path to dataset needed for psychlab_*, see https://github.com/deepmind/lab/tree/master/data/brady_konkle_oliva2008
        config['datasetPath'] = ''
        # Labyrinth homepath.
        # deepmind_lab.set_runfiles_path(homepath)

        self._num_action_repeats = num_action_repeats
        self._random_state = np.random.RandomState(seed=seed)

        self._env = deepmind_lab.Lab(
            level=game,
            observations=['RGB_INTERLEAVED'],
            level_cache=LevelCache(level_cache_dir) if level_cache_dir else None,
            config={k: str(v) for k, v in config.items()},
        )
        self._action_set = action_set
        self.action_space = gym.spaces.Discrete(len(self._action_set))
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(config['height'], config['width'], 3),
            dtype=np.uint8)

    def _observation(self):
        return self._env.observations()['RGB_INTERLEAVED']

    def reset(self):
        self._env.reset(seed=self._random_state.randint(0, 2 ** 31 - 1))
        return self._observation()

    def step(self, action):
        raw_action = np.array(self._action_set[action], np.intc)
        reward = self._env.step(raw_action, num_steps=self._num_action_repeats)
        done = not self._env.is_running()
        if not done:
            observation = self._observation()
        else:
            # Do not have actual observation in done state, but need to return something
            observation = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        return observation, reward, done, {}

    def close(self):
        self._env.close()


class LevelCache(object):
    """Level cache."""

    def __init__(self, cache_dir):
        self._cache_dir = cache_dir

    def get_path(self, key):
        key = hashlib.md5(key.encode('utf-8')).hexdigest()
        dir_, filename = key[:3], key[3:]
        return os.path.join(self._cache_dir, dir_, filename)

    def fetch(self, key, pk3_path):
        path = self.get_path(key)
        raise NotImplementedError
        # try:
        #   tf.io.gfile.copy(path, pk3_path, overwrite=True)
        #   return True
        # except tf.errors.OpError:
        #   return False

    def write(self, key, pk3_path):
        path = self.get_path(key)
        raise NotImplementedError
        # if not tf.io.gfile.exists(path):
        #   tf.io.gfile.makedirs(os.path.dirname(path))
        #   tf.io.gfile.copy(pk3_path, path)
