from typing import Tuple

import gym
import gym.spaces
import gym_minigrid
import gym_minigrid.envs
import gym_minigrid.minigrid
from gym_minigrid.minigrid import COLOR_TO_IDX, OBJECT_TO_IDX
import numpy as np


class MiniGrid(gym.Env):

    GRID_VALUES = np.array([  # shape=(33,3)
        [0, 0, 0],  # Invisible
        [1, 0, 0],  # Empty
        [2, 5, 0],  # Wall
        [8, 1, 0],  # Goal
        # Agent
        [10, 0, 0],
        [10, 0, 1],
        [10, 0, 2],
        [10, 0, 3],
        # Door (color, state)
        [4, 0, 0],
        [4, 0, 1],
        [4, 1, 0],
        [4, 1, 1],
        [4, 2, 0],
        [4, 2, 1],
        [4, 3, 0],
        [4, 3, 1],
        [4, 4, 0],
        [4, 4, 1],
        [4, 5, 0],
        [4, 5, 1],
        # Key (color)
        [5, 0, 0],
        [5, 1, 0],
        [5, 2, 0],
        [5, 3, 0],
        [5, 4, 0],
        [5, 5, 0],
        # Ball (color)
        [6, 0, 0],
        [6, 1, 0],
        [6, 2, 0],
        [6, 3, 0],
        [6, 4, 0],
        [6, 5, 0],
        # Box (color)
        [7, 0, 0],
        [7, 1, 0],
        [7, 2, 0],
        [7, 3, 0],
        [7, 4, 0],
        [7, 5, 0],
    ])

    def __init__(self, env_name, max_steps=1000, seed=None, agent_init_pos=None, agent_init_dir=0):
        env = gym.make(env_name)
        assert isinstance(env, gym_minigrid.envs.MiniGridEnv)
        self.env = env
        self.env.max_steps = max_steps
        if seed:
            self.env.seed(seed)
        self.max_steps = max_steps
        self.agent_init_pos = agent_init_pos
        self.agent_init_dir = agent_init_dir

        grid = self.env.grid.encode()  # type: ignore  # Grid is already generated when env is created
        self.map_size = n = grid.shape[0]
        self.map_centered_size = m = 2 * n - 3  # 11x11 => 19x19

        spaces = {}
        spaces['image'] = gym.spaces.Box(0, 255, (7, 7), np.uint8)
        spaces['map'] = gym.spaces.Box(0, 255, (n, n), np.uint8)
        spaces['map_agent'] = gym.spaces.Box(0, 255, (n, n), np.uint8)
        spaces['map_masked'] = gym.spaces.Box(0, 255, (n, n), np.uint8)
        spaces['map_vis'] = gym.spaces.Box(0, self.max_steps, (n, n), np.uint16)
        spaces['map_centered'] = gym.spaces.Box(0, 255, (m, m), np.uint8)
        self.observation_space = gym.spaces.Dict(spaces)
        self.action_space = self.env.action_space

        self.map_last_seen = np.zeros(grid.shape[0:2], dtype=np.uint16)

    @staticmethod
    def has_action_repeat():
        return False

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info

    def reset(self):
        obs = self.env.reset()
        if self.agent_init_pos:
            # Initialize agent in a fixed position, so it can build a map
            self.env.agent_pos = np.array(self.agent_init_pos)
            self.env.agent_dir = self.agent_init_dir
            self.env.grid.set(*self.env.agent_pos, None)  # type: ignore  # Remove if something was there
            obs = self.env.gen_obs()
        self.reset_map_last_seen()
        return self.observation(obs)

    def observation(self, obs_in):
        img = obs_in['image']

        obs = {}
        obs['image'] = self.to_categorical(img)
        obs['map'] = self.to_categorical(self.map(with_agent=False))
        obs['map_agent'] = self.to_categorical(self.map(with_agent=True))
        vis_mask = self.global_vis_mask(img)
        obs['map_masked'] = obs['map_agent'] * vis_mask
        obs['map_vis'] = self.update_map_last_seen(vis_mask)
        obs['map_centered'] = self.to_categorical(self.map_centered())

        for k in obs:
            assert obs[k].shape == self.observation_space[k].shape, f"Wrong shape {k}: {obs[k].shape} != {self.observation_space[k].shape}"  # type: ignore
        return obs

    @staticmethod
    def to_categorical(image_ids):
        n = len(MiniGrid.GRID_VALUES)
        out = np.zeros(image_ids.shape[:-1] + (n,))
        for i in range(n):
            val = MiniGrid.GRID_VALUES[i]
            out[..., i] = (image_ids == val).all(axis=-1)
        out = out.argmax(axis=-1).astype(np.uint8)  # (..., 7, 7, 33) => (..., 7, 7)
        return out

    @staticmethod
    def from_categorical(img):
        return MiniGrid.GRID_VALUES[img]

    def map(self, with_agent=True):
        out = self.env.grid.encode()  # type: ignore
        if with_agent:
            out[self.env.agent_pos[0]][self.env.agent_pos[1]] = np.array([  # type: ignore
                OBJECT_TO_IDX['agent'],
                COLOR_TO_IDX['red'],
                self.env.agent_dir
            ])
        return out

    def map_centered(self):
        n = self.map_centered_size
        x, y = self.env.agent_pos  # type: ignore
        grid = self.env.grid.slice(x - (n - 1) // 2, y - (n - 1) // 2, n, n)  # type: ignore
        for i in range(self.env.agent_dir + 1):  # type: ignore
            grid = grid.rotate_left()
        image = grid.encode()
        return image

    def reset_map_last_seen(self):
        self.map_last_seen *= 0
        self.map_last_seen += self.max_steps

    def update_map_last_seen(self, map_vis):
        # Update how long ago each map grid was seen. If not seen, then set to max_steps
        self.map_last_seen += 1
        np.clip(self.map_last_seen, 0, self.max_steps, out=self.map_last_seen)
        self.map_last_seen *= (~map_vis)
        return self.map_last_seen.copy()

    def global_vis_mask(self, img):
        # Mark which cells are visible on the global map
        obs_vis_mask = img[:, :, 0] > 0
        glb_vis_mask = np.zeros((self.env.width, self.env.height), dtype=np.bool)
        x, y, mask = self.obs_global_coords()
        glb_vis_mask[x[mask], y[mask]] = obs_vis_mask[mask]
        return glb_vis_mask

    def obs_global_coords(self):
        n = self.env.agent_view_size  # =7
        x = np.zeros((n, n), int)
        y = np.zeros((n, n), int)
        mask = np.zeros((n, n), bool)

        # Transform from local to global coordinates
        # TODO perf: do without loops
        f_vec = self.env.dir_vec
        r_vec = self.env.right_vec
        top_left = self.env.agent_pos + f_vec * (n - 1) - r_vec * (n // 2)
        for vis_j in range(0, n):
            for vis_i in range(0, n):
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)
                mask[vis_i, vis_j] = (abs_i >= 0 and abs_i < self.env.width and abs_j >= 0 and abs_j < self.env.height)
                if mask[vis_i, vis_j]:
                    x[vis_i, vis_j] = abs_i
                    y[vis_i, vis_j] = abs_j

        return x, y, mask

    @staticmethod
    def render_map(map_, tile_size=16):
        map_ = MiniGrid.from_categorical(map_)

        # Find and remove special "agent" object
        agent_pos, agent_dir = None, None
        x, y = (map_[:, :, 0] == OBJECT_TO_IDX['agent']).nonzero()
        if len(x) > 0:
            x, y = x[0], y[0]
            agent_pos = x, y
            agent_dir = map_[x][y][2]
            map_[x][y] = np.array([1, 0, 0])  # EMPTY

        grid, vis_mask = gym_minigrid.minigrid.Grid.decode(map_)
        img = grid.render(tile_size, agent_pos=agent_pos, agent_dir=agent_dir, highlight_mask=~vis_mask)
        return img

    def close(self):
        pass


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
