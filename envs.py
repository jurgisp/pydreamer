import numpy as np
import gym
import gym_minigrid
import gym_minigrid.envs
import torch


class MiniGrid:

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

    def __init__(self, env_name, max_steps=1000, seed=1337, agent_init_pos=None, agent_init_dir=0):
        self._env = gym.make(env_name)
        self._env.max_steps = max_steps
        self._env.seed(seed)
        self.max_steps = max_steps
        self.agent_init_pos = agent_init_pos
        self.agent_init_dir = agent_init_dir

        grid = self._env.grid.encode()  # Grid is already generated when env is created
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
        self.action_space = self._env.action_space

        self._map_last_seen = np.zeros(grid.shape[0:2], dtype=np.uint16)

    @staticmethod
    def has_action_repeat():
        return False

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return self._observation(obs), reward, done, info

    def reset(self):
        obs = self._env.reset()
        if self.agent_init_pos:
            # Initialize agent in a fixed position, so it can build a map
            self._env.agent_pos = np.array(self.agent_init_pos)
            self._env.agent_dir = self.agent_init_dir
            self._env.grid.set(*self._env.agent_pos, None)  # Remove if something was there
            obs = self._env.gen_obs()
        self._reset_map_last_seen()
        return self._observation(obs)

    def _observation(self, obs_in):
        img = obs_in['image']

        obs = {}
        obs['image'] = self.to_categorical(img)
        obs['map'] = self.to_categorical(self._map(with_agent=False))
        obs['map_agent'] = self.to_categorical(self._map(with_agent=True))
        vis_mask = self._global_vis_mask(img)
        obs['map_masked'] = obs['map_agent'] * vis_mask
        obs['map_vis'] = self._update_map_last_seen(vis_mask)
        obs['map_centered'] = self.to_categorical(self._map_centered())

        for k in obs:
            assert obs[k].shape == self.observation_space[k].shape, f"Wrong shape {k}: {obs[k].shape} != {self.observation_space[k].shape}"
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

    def _map(self, with_agent=True):
        out = self._env.grid.encode()
        if with_agent:
            out[self._env.agent_pos[0]][self._env.agent_pos[1]] = np.array([
                gym_minigrid.minigrid.OBJECT_TO_IDX['agent'],
                gym_minigrid.minigrid.COLOR_TO_IDX['red'],
                self._env.agent_dir
            ])
        return out

    def _map_centered(self):
        n = self.map_centered_size
        x, y = self._env.agent_pos
        grid = self._env.grid.slice(x - (n - 1) // 2, y - (n - 1) // 2, n, n)
        for i in range(self._env.agent_dir + 1):
            grid = grid.rotate_left()
        image = grid.encode()
        return image

    def _reset_map_last_seen(self):
        self._map_last_seen *= 0
        self._map_last_seen += self.max_steps

    def _update_map_last_seen(self, map_vis):
        # Update how long ago each map grid was seen. If not seen, then set to max_steps
        self._map_last_seen += 1
        np.clip(self._map_last_seen, 0, self.max_steps, out=self._map_last_seen)
        self._map_last_seen *= (~map_vis)
        return self._map_last_seen.copy()

    def _global_vis_mask(self, img):
        # Mark which cells are visible on the global map
        obs_vis_mask = img[:, :, 0] > 0
        glb_vis_mask = np.zeros((self._env.width, self._env.height), dtype=np.bool)
        x, y, mask = self._obs_global_coords()
        glb_vis_mask[x[mask], y[mask]] = obs_vis_mask[mask]
        return glb_vis_mask

    def _obs_global_coords(self):
        n = self._env.agent_view_size  # =7
        x = np.zeros((n, n), int)
        y = np.zeros((n, n), int)
        mask = np.zeros((n, n), bool)

        # Transform from local to global coordinates
        # TODO perf: do without loops
        f_vec = self._env.dir_vec
        r_vec = self._env.right_vec
        top_left = self._env.agent_pos + f_vec * (n - 1) - r_vec * (n // 2)
        for vis_j in range(0, n):
            for vis_i in range(0, n):
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)
                mask[vis_i, vis_j] = (abs_i >= 0 and abs_i < self._env.width and abs_j >= 0 and abs_j < self._env.height)
                if mask[vis_i, vis_j]:
                    x[vis_i, vis_j] = abs_i
                    y[vis_i, vis_j] = abs_j

        return x, y, mask

    @staticmethod
    def render_map(map_, tile_size=16):
        map_ = MiniGrid.from_categorical(map_)

        # Find and remove special "agent" object
        agent_pos, agent_dir = None, None
        x, y = (map_[:, :, 0] == gym_minigrid.minigrid.OBJECT_TO_IDX['agent']).nonzero()
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


def worldgrid_map_accuracy(output, target, agent_pos, agent_dir):
    assert len(output.shape) == 4
    assert len(target.shape) == 4
    assert len(agent_pos.shape) == 2 and agent_pos.shape[-1] == 2
    assert len(agent_dir.shape) == 2 and agent_dir.shape[-1] == 2

    val_path = np.array([103, 103, 105]) / 255.0 - 0.5
    val_wall = np.array([64, 209, 255]) / 255.0 - 0.5

    def diff_val(x, val):
        x = x.clone()  # (B,C,H,W)
        x[:, 0, :, :] -= val[0]
        x[:, 1, :, :] -= val[1]
        x[:, 2, :, :] -= val[2]
        return torch.square(x).sum(dim=1)

    B = output.shape[0]
    ax = agent_pos[:, 0]
    ay = agent_pos[:, 1]
    dx = agent_dir[:, 0]
    dy = agent_dir[:, 1]
    idx = torch.arange(0, B, 1, device=target.device)

    target_diff_path = diff_val(target, val_path)
    target_diff_wall = diff_val(target, val_wall)
    output_diff_path = diff_val(output, val_path)
    output_diff_wall = diff_val(output, val_wall)
    target_wall = target_diff_wall < target_diff_path
    output_wall = output_diff_wall < output_diff_path
    correct = output_wall == target_wall

    check_mask = torch.zeros_like(target_wall)
    for iy in range(9):
        for ix in range(9):
            # (x,y) - global coords
            # (ax,ay) - agent coords
            # (dx,dy) - agent direction vector
            # (mx,my) - agent-centric map coords
            # (mx-mx0)/res = -dy(x-ax) + dx(y-ay)
            # (my-my0)/res = -dx(x-ax) - dy(y-ay)
            x = (ix + 0.5) / 4.5 - 1.0
            y = (iy + 0.5) / 4.5 - 1.0
            mx = torch.round(31.5 + 32 * (- dy * (x - ax) + dx * (y - ay)) / 2.0).to(torch.long)
            my = torch.round(31.5 + 32 * (- dx * (x - ax) - dy * (y - ay)) / 2.0).to(torch.long)
            mask = (mx >= 0) & (mx < 64) & (my >= 0) & (my < 64)
            check_mask[idx[mask], my[mask], mx[mask]] = True

    acc = (correct & check_mask).sum(dim=[1, 2]) / check_mask.sum(dim=[1, 2])
    mask = ((-1e-3 < dy) & (dy < 1e-3)) | ((-1e-3 < dx) & (dx < 1e-3))  # calculate only at right angles
    acc[~mask] = np.nan

    return acc
