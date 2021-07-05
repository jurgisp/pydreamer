import argparse
import numpy as np
import datetime
import time
import pathlib
import gym
from envs import MiniGrid
import mlflow

import tools


def main(output_dir,
         env_name,
         policy,
         conf,
         ):
    # delete_every = 100  # if conf.delete_old is set

    if conf.save_to_mlflow:
        run = mlflow.start_run(run_name=f'{env_name}-s{conf.seed}')
        print(f'Mlflow run {run.info.run_id} in experiment {run.info.experiment_id}')
    else:
        output_dir = pathlib.Path(output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

    if env_name.startswith('MiniGrid-'):
        env = MiniGrid(env_name, max_steps=conf.max_steps, seed=conf.seed)

    elif env_name.startswith('MiniWorld-'):
        import gym_miniworld
        from gym_miniworld.wrappers import DictWrapper, MapWrapper, AgentPosWrapper
        env = gym.make(env_name, max_steps=conf.max_steps)
        env = DictWrapper(env)
        env = MapWrapper(env)
        env = AgentPosWrapper(env)

    else:
        env = gym.make(env_name, max_steps=conf.max_steps)

    env = CollectWrapper(env, conf.max_steps)

    if policy == 'random':
        policy = RandomPolicy(env.action_space)
    elif policy == 'minigrid_wander':
        policy = MinigridWanderPolicy()
    elif policy == 'maze_bouncing_ball':
        policy = MazeBouncingBallPolicy()
    else:
        assert False, 'Unknown policy'

    steps, episodes = 0, 0
    datas = []
    visited_stats = []

    while steps < conf.num_steps:

        # Unroll one episode

        obs, done = env.reset(), False
        epsteps, timer = 0, time.time()
        while not done:
            action = policy(obs)
            obs, reward, done, info = env.step(action)
            steps += 1
            epsteps += 1
        data = info['episode']  # type: ignore

        # Calculate visited

        agent_pos = data['agent_pos']
        agent_pos = np.floor(agent_pos / 3 / 2)
        agent_pos_visited = len(np.unique(agent_pos, axis=0))
        visited_pct = agent_pos_visited / 25
        visited_stats.append(visited_pct)

        # Log

        fps = epsteps / (time.time() - timer + 1e-6)
        if episodes == 0:
            print('Data sample: ', {k: v.shape for k, v in data.items()})

        print(f"[{steps:08}/{conf.num_steps:08}] "
              f"Episode {episodes} recorded:"
              f"  steps: {epsteps}"
              f",  reward: {data['reward'].sum()}"
              f",  explored%: {visited_pct:.1%}|{np.mean(visited_stats):.1%}"
              f",  fps: {fps:.0f}"
              )

        # Save to npz

        episodes += 1
        datas.append(data)

        if len(datas) == conf.episodes_per_npz:

            # Concatenate 10 episodes

            data = {}
            for key in datas[0]:
                data[key] = np.concatenate([b[key] for b in datas], axis=0)
            datas = []
            assert data['reset'].sum() == conf.episodes_per_npz

            # NHWC => HWCN for better compression

            data['image_t'] = data['image'].transpose(1, 2, 3, 0)
            del data['image']

            # Save to npz

            if episodes == conf.episodes_per_npz:
                print('Data sample: ', {k: v.shape for k, v in data.items()})

            n = data['reset'].shape[0] - 1
            if conf.episodes_per_npz > 1:
                fname = f's{conf.seed}-ep{episodes-conf.episodes_per_npz:06}_{episodes-1:06}-{n:04}.npz'
            else:
                fname = f's{conf.seed}-ep{episodes-1:06}-{n:04}.npz'

            if conf.save_to_mlflow:
                tools.mlflow_log_npz(data, fname, 'episodes', verbose=True)
            else:
                fname = output_dir / fname
                tools.save_npz(data, fname)

        # Delete old

        # if conf.delete_old and episodes % delete_every == 0:
        #     assert not conf.save_to_mlflow
        #     for i_new in range(episodes - delete_every + 1, episodes + 1):
        #         i_old = i_new - conf.delete_old
        #         if i_old < 0:
        #             continue
        #         del_fname = output_dir / f's{conf.seed}-ep{i_old:06}-{epsteps:04}.npz'
        #         print(f'Deleting {del_fname}')
        #         del_fname.unlink()

        # if conf.sleep:
        #     time.sleep(conf.sleep)


class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def __call__(self, obs):
        return self.action_space.sample()


class MinigridWanderPolicy:
    def __call__(self, obs):
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
            return 0

        # Door on right => turn with 50%
        if right[0] == 4 and np.random.rand() < 0.50:
            return 1

        # Empty left  => turn with 10%
        if left[0] in empty and np.random.rand() < 0.10:
            return 0

        # Empty right => turn with 10%
        if right[0] in empty and np.random.rand() < 0.10:
            return 1

        # Closed door => open
        if front[0] == 4 and front[2] == 1:
            return 5

        # Empty or open door => forward
        if front[0] in empty or (front[0] == 4 and front[2] == 0):
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


class MazeBouncingBallPolicy:
    # Policy:
    #   1) Forward until you hit a wall
    #   2) Turn in random 360 direction
    #   3) Go to 1)

    def __init__(self):
        self.pos = None
        self.turns_remaining = 0

    def __call__(self, obs):
        assert 'agent_pos' in obs, f'Need agent position'
        pos = obs['agent_pos']
        action = -1

        # print(f'{self.pos} => {pos} ({obs["agent_dir"]})')

        if self.turns_remaining == 0:
            if self.pos is None or not np.all(self.pos == pos):
                # Going forward
                action = 2
                self.pos = pos
            else:
                # Hit the wall - start turning
                if np.random.randint(2) == 0:
                    self.turns_remaining = -np.random.randint(2, 5)  # Left
                else:
                    self.turns_remaining = np.random.randint(2, 5)  # Right
                self.pos = None

        if self.turns_remaining > 0:
            # Turning right
            action = 1
            self.turns_remaining -= 1

        elif self.turns_remaining < 0:
            # Turning left
            action = 0
            self.turns_remaining += 1

        assert action >= 0
        return action


class CollectWrapper:

    def __init__(self, env, max_steps):
        self._env = env
        self._episode = []
        self._max_steps = max_steps
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
        transition['terminal'] = done if len(self._episode) < self._max_steps else False  # Only True if actual terminal state, not done because of max_steps
        transition['reset'] = False

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
        transition['terminal'] = False
        transition['reset'] = True
        self._episode = [transition]
        return obs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env')
    parser.add_argument('policy')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--num_steps', type=int, default=1_000_000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--delete_old', type=int, default=0)
    parser.add_argument('--sleep', type=int, default=0)
    parser.add_argument('--save_to_mlflow', action='store_true')
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--episodes_per_npz', type=int, default=1)
    args = parser.parse_args()

    output_dir = args.output_dir or f"data/{args.env}/{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"

    main(output_dir, args.env, args.policy, args)
