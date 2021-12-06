# Generate training and test data with TRPO algorithm
# https://stable-baselines.readthedocs.io/en/master/modules/trpo.html
import gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import h5py as hdf
import pickle as pkl
from src.utils import Experience, ReplayBuffer

ENV_CONFIGS = {
    "bandit":
        {"generation": {
            "n_steps": 128,
            "n_steps_total": 5000,
            "num_of_tasks": 15,
            "observation_space": 1,
            "action_space": 1
        }
        },
    "ant_dir":
        {"generation": {
            "n_steps": 128,
            "n_steps_total": 5000,
            "num_of_tasks": 15,
            "observation_space": 1,
            "action_space": 1
        }
        }
}


def generate_data(ENV: str):
    config = ENV_CONFIGS[ENV]['generation']
    n_steps_total = config['n_steps_total']
    for i in tqdm(range(config['num_of_tasks'])):
        env = load_env(ENV)
        task_params = [{'p_dist': env.p_dist}]
        replay_buffer = ReplayBuffer(n_steps_total, config['observation_space'], 10)
        task = generate_trajectories(env, config['n_steps'], n_steps_total)
        replay_buffer.add_trajectory(task)
        env.close()
        replay_buffer.save(f'../data/buffers_bandits_train_{i}.hdf5')
        with open(f'../data/env_bandits_train_task{i}.pkl', 'wb') as f:
            pkl.dump(task_params, f)
            f.close()


def generate_trajectories(env: gym.Env, n_steps: int, n_steps_total: int):
    model = PPO('MlpPolicy', env, verbose=0, n_steps=n_steps)
    callback = RolloutCallback()
    model.learn(n_steps_total, callback=callback)
    return callback.get_trajectory()


class RolloutCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RolloutCallback, self).__init__(verbose)
        self.trajectory = []

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        data = self.locals
        action = np.zeros(10)
        action[data['actions']] = 1
        state, action, next_state, reward, done = data['obs_tensor'].cpu().numpy(), action, data['new_obs'], \
                                                  data['rewards'], data['dones']
        self.trajectory.append(Experience(state, action, next_state, reward, done))
        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        pass

    def get_trajectory(self):
        return self.trajectory
