# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import json
import os
import random
import time
from logging import debug, info, warning
from typing import NamedTuple

import dm_env
import grpc
import gym
import gym.spaces
import numpy as np
from dm_env_rpc.v1 import connection as dm_env_rpc_connection
from dm_env_rpc.v1 import dm_env_adaptor, dm_env_rpc_pb2
from dm_env_rpc.v1 import error as dm_env_rpc_error
from dm_env_rpc.v1 import tensor_utils
from PIL import Image

ACTION_SET = [
    {'MOVE_BACK_FORWARD': 0, 'STRAFE_LEFT_RIGHT': 0, 'LOOK_LEFT_RIGHT': 0, 'LOOK_DOWN_UP': 0},
    {'MOVE_BACK_FORWARD': +1, 'STRAFE_LEFT_RIGHT': 0, 'LOOK_LEFT_RIGHT': 0, 'LOOK_DOWN_UP': 0},
    {'MOVE_BACK_FORWARD': -1, 'STRAFE_LEFT_RIGHT': 0, 'LOOK_LEFT_RIGHT': 0, 'LOOK_DOWN_UP': 0},
    {'MOVE_BACK_FORWARD': 0, 'STRAFE_LEFT_RIGHT': +1, 'LOOK_LEFT_RIGHT': 0, 'LOOK_DOWN_UP': 0},
    {'MOVE_BACK_FORWARD': 0, 'STRAFE_LEFT_RIGHT': -1, 'LOOK_LEFT_RIGHT': 0, 'LOOK_DOWN_UP': 0},
    {'MOVE_BACK_FORWARD': 0, 'STRAFE_LEFT_RIGHT': 0, 'LOOK_LEFT_RIGHT': +1, 'LOOK_DOWN_UP': 0},
    {'MOVE_BACK_FORWARD': 0, 'STRAFE_LEFT_RIGHT': 0, 'LOOK_LEFT_RIGHT': -1, 'LOOK_DOWN_UP': 0},
    # {'MOVE_BACK_FORWARD': 0, 'STRAFE_LEFT_RIGHT': 0, 'LOOK_LEFT_RIGHT': 0, 'LOOK_DOWN_UP': +1},
    # {'MOVE_BACK_FORWARD': 0, 'STRAFE_LEFT_RIGHT': 0, 'LOOK_LEFT_RIGHT': 0, 'LOOK_DOWN_UP': -1},
    {'MOVE_BACK_FORWARD': +1, 'STRAFE_LEFT_RIGHT': 0, 'LOOK_LEFT_RIGHT': +1, 'LOOK_DOWN_UP': 0},
    {'MOVE_BACK_FORWARD': +1, 'STRAFE_LEFT_RIGHT': 0, 'LOOK_LEFT_RIGHT': -1, 'LOOK_DOWN_UP': 0},
]

_MEMORY_TASK_LEVEL_NAMES = [
    'spot_diff_motion_train',
    'spot_diff_multi_train',
    'spot_diff_passive_train',
    'spot_diff_train',
    'invisible_goal_empty_arena_train',
    'invisible_goal_with_buildings_train',
    'visible_goal_with_buildings_train',
    'transitive_inference_train_small', 'transitive_inference_train_large',
    # There are also levels:
    #   *_interpolate
    #   *_extrapolate
    #   *_holdout_small
    #   *_holdout_large
    #   *_holdout_interpolate
    #   *_holdout_extrapolate
]


class DMMEnv(gym.Env):
    """DM Memory Tasks environment.

    Task descriptions: 
        - https://sites.google.com/view/memory-tasks-suite/home#h.p_abzaAYKn6eel
    Wrapper code adapted from: 
        - https://github.com/deepmind/dm_memorytasks
    """

    def __init__(self,
                 level_name,
                 num_action_repeats=1,
                 action_set=ACTION_SET,
                 address="localhost:2222",
                 worker_id=0,
                 ):
        # Get worker address which hosts environment from TF_CONFIG
        tf_config = os.environ.get('TF_CONFIG', None)
        if tf_config:
            debug(f"TF_CONFIG is set: {tf_config}")
            tf_config = json.loads(tf_config)
            assert worker_id < len(tf_config['cluster']['worker']), f"Not enough workers for worker_id={worker_id}"
            address = tf_config['cluster']['worker'][worker_id]

        settings = EnvironmentSettings(level_name=level_name, seed=random.randint(1, 999999))
        channel, connection, specs = _connect_to_environment(settings, address)
        requested_observations = ['RGB_INTERLEAVED']  # Also available: ['AvatarPosition', 'Score']
        self._rpc_env = dm_env_adaptor.DmEnvAdaptor(connection, specs, requested_observations)
        self._channel = channel

        self._num_action_repeats = num_action_repeats
        self._action_set = action_set
        self.action_space = gym.spaces.Discrete(len(self._action_set))
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

    def reset(self):
        timestep = self._rpc_env.reset()
        return self.observation(timestep)

    def step(self, action):
        timestep = None
        reward = 0.0

        for _ in range(self._num_action_repeats):
            timestep = self._rpc_env.step(self._action_set[action])
            reward += timestep.reward or 0.
            if timestep.last():
                break

        assert timestep is not None
        obs = self.observation(timestep)
        done = timestep.last()
        # NOTE: timestep.discount should distinguish terminal from time limit, but doesn't seem implemented in DMM
        # is_terminal = timestep.discount == 0
        return obs, reward, done, {}

    def observation(self, timestep: dm_env.TimeStep):
        img = timestep.observation['RGB_INTERLEAVED']
        img = np.array(Image.fromarray(img).resize((64, 64), Image.NEAREST))
        return img

    def close(self):
        self._rpc_env.close()
        self._channel.close()


class EnvironmentSettings(NamedTuple):
    """Collection of settings used to start a specific Memory task.

      Required attributes:
        seed: Seed to initialize the environment's RNG.
        level_name: Name of the level to load.
      Optional attributes:
        width: Width (in pixels) of the desired RGB observation; defaults to 96.
        height: Height (in pixels) of the desired RGB observation; defaults to 72.
        episode_length_seconds: Maximum episode length (in seconds); defaults to
          120.
        num_action_repeats: Number of times to step the environment with the
          provided action in calls to `step()`.
    """
    seed: int
    level_name: str
    width: int = 96
    height: int = 72
    episode_length_seconds: float = 120.0
    num_action_repeats: int = 1


def _connect_to_environment(settings: EnvironmentSettings, address: str):
    info(f'Connecting to remote DMM env on {address}...')
    channel, connection = _create_channel_and_connection(address)
    info('Connected to remote DMM env.')

    # original_send = connection.send
    # connection.send = lambda request: _wrap_send(lambda: original_send(request))

    info(f'Initializing remote DMM env {settings.level_name} ({settings.seed})...')

    world_name = connection.send(
        dm_env_rpc_pb2.CreateWorldRequest(settings={
            'seed': tensor_utils.pack_tensor(settings.seed),
            'episodeId': tensor_utils.pack_tensor(0),
            'levelName': tensor_utils.pack_tensor(settings.level_name),
        })).world_name

    specs = connection.send(
        dm_env_rpc_pb2.JoinWorldRequest(
            world_name=world_name,
            settings={
                'width': tensor_utils.pack_tensor(settings.width),
                'height': tensor_utils.pack_tensor(settings.height),
                'EpisodeLengthSeconds': tensor_utils.pack_tensor(settings.episode_length_seconds),
            }
        )).specs

    # debug(f'DMM env specs: {specs}')
    info('Initialized remote DMM env.')
    return channel, connection, specs


def _create_channel_and_connection(address: str, max_attempts: int = 10):
    """Returns a tuple of `(channel, connection)`."""
    for _ in range(max_attempts):
        debug('GRPC creating channel...')
        channel = grpc.insecure_channel(address)  # host:port
        _check_grpc_channel_ready(channel)
        debug('GRPC creating connection...')
        connection = dm_env_rpc_connection.Connection(channel)
        if _can_send_message(connection):
            # CONNECTED
            return channel, connection
        else:
            warning(f'GRPC problem connecting to {address} - will retry {max_attempts} times')
            connection.close()
            channel.close()
            time.sleep(1.0)

    raise Exception(f'Could not connect to DMM env on {address}')


def _check_grpc_channel_ready(channel, max_attempts: int = 10):
    for _ in range(max_attempts - 1):
        try:
            debug('GRPC checking channel...')
            return grpc.channel_ready_future(channel).result(timeout=1)
        except grpc.FutureTimeoutError:
            warning('GRPC checking channel - failed.')
            pass
    return grpc.channel_ready_future(channel).result(timeout=1)


def _can_send_message(connection):
    try:
        debug('GRPC trying send message...')
        connection.send(dm_env_rpc_pb2.StepRequest())
    except dm_env_rpc_error.DmEnvRpcError:
        debug('GRPC send message - OK.')
        return True
    except grpc.RpcError:
        warning('GRPC send message - failed.')
        return False
