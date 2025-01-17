import numpy as np
from typing import List
from src.tp_envs.half_cheetah_dir import HalfCheetahDirEnv as HalfCheetahDirEnv_
from src.tp_envs.ant_dir import AntDirEnv as AntDirEnv_
import gym
import gym.spaces as spaces
from gym.utils import seeding


class BanditEnv(gym.Env):

    def __init__(self, tasks: List[dict], n_tasks: int):
        super(BanditEnv, self).__init__()
        if tasks is None:
            assert n_tasks is not None, "Either tasks or n_tasks must be non-None"
            tasks = self.sample_tasks(n_tasks)
        self.tasks = tasks
        self.set_task_idx(0)
        self._max_episode_steps = 1000
        self.k = len(self._task['p_dist'])
        self.action_space = spaces.Discrete(self.k)
        self.observation_space = spaces.Box(low=1, high=1,
                                            shape=(1,), dtype=np.float32)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_tasks(self, num_tasks):
        means = self.np_random.rand(num_tasks, self.k)
        tasks = [{'p': mean} for mean in means]
        return tasks

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._mean = self._task['p_dist']
        self.reset()

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def set_task(self, task):
        self._task = task
        self._mean = self._task['p_dist']
        self.reset()

    def reset(self):
        return np.ones(1, dtype=np.float32)

    def step(self, action):
        assert self.action_space.contains(action)
        mean = self._mean[action]
        reward = self.np_random.binomial(1, mean)
        observation = np.ones(1, dtype=np.float32)

        return observation, reward, False, {'task': self._task}

    def set_task_idx(self, idx):
        self.set_task(self.tasks[idx])


class HalfCheetahDirEnv(HalfCheetahDirEnv_):
    def __init__(self, tasks: List[dict], include_goal: bool = False):
        self.include_goal = include_goal
        super(HalfCheetahDirEnv, self).__init__()
        if tasks is None:
            tasks = [{'direction': 1}, {'direction': -1}]
        self.tasks = tasks
        self.set_task_idx(0)
        self._max_episode_steps = 200

    def _get_obs(self):
        if self.include_goal:
            idx = 0
            try:
                idx = self.tasks.index(self._task)
            except:
                pass
            one_hot = np.zeros(len(self.tasks), dtype=np.float32)
            one_hot[idx] = 1.0
            obs = super()._get_obs()
            obs = np.concatenate([obs, one_hot])
        else:
            obs = super()._get_obs()
        return obs

    def set_task(self, task):
        self._task = task
        self._goal_dir = self._task['direction']
        self.reset()

    def set_task_idx(self, idx):
        self.set_task(self.tasks[idx])


class AntDirEnv(AntDirEnv_):
    def __init__(self, tasks: List[dict], n_tasks: int = None, include_goal: bool = False):
        self.include_goal = include_goal
        super(AntDirEnv, self).__init__(forward_backward=n_tasks == 2)
        if tasks is None:
            assert n_tasks is not None, "Either tasks or n_tasks must be non-None"
            tasks = self.sample_tasks(n_tasks)
        self.tasks = tasks
        self.n_tasks = len(self.tasks)
        self.set_task_idx(0)
        self._max_episode_steps = 200

    def _get_obs(self):
        if self.include_goal:
            idx = 0
            try:
                idx = self.tasks.index(self._task)
            except:
                pass
            one_hot = np.zeros(50, dtype=np.float32)
            one_hot[idx] = 1.0
            obs = super()._get_obs()
            obs = np.concatenate([obs, one_hot])
        else:
            obs = super()._get_obs()
        return obs

    def set_task(self, task):
        self._task = task
        self._goal = task['goal']
        self.reset()

    def set_task_idx(self, idx):
        self.set_task(self.tasks[idx])
