import argparse
from multiprocessing import Process, set_start_method

import gym
import numpy as np
from gym.spaces import Discrete
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from src.envs import BanditEnv, AntDirEnv
import pickle as pkl

from src.macaw import env_action_dim
from src.utils import Experience, ReplayBuffer

ENV_CONFIGS = {
    "bandit": {
        "n_steps_total": 5000,
    },
    "ant_dir": {
        "n_steps_total": 1000000,
    }
}


def generate_data(env_name: str, num_tasks: int, start_index: int, type=int):
    if env_name == 'bandit':
        tasks = [{'p_dist': np.random.uniform(size=10)} for _ in range(num_tasks)]
        env = BanditEnv(tasks, num_tasks)
    elif env_name == 'ant_dir':
        if type == 1:
            tasks = [{'goal': np.random.uniform(0, np.pi / 2, size=(1,))[0]} for _ in range(num_tasks)]
        elif type == 2:
            tasks = [{'goal': np.random.uniform(0, np.pi, size=(1,))[0]} for _ in range(num_tasks)]
        elif type == 3:
            tasks = [{'goal': np.random.uniform(0, (np.pi / 2 + np.pi), size=(1,))[0]} for _ in range(num_tasks)]
        elif type == 4:
            tasks = [{'goal': np.random.uniform(0, 2 * np.pi, size=(1,))[0]} for _ in range(num_tasks)]
        env = AntDirEnv(tasks, num_tasks)

    config = ENV_CONFIGS[env_name]
    observation_dim = env.observation_space.shape[0]
    action_dim = env_action_dim(env)
    for i, task in enumerate(tasks):
        task_params = [task]
        env.set_task_idx(i)
        replay_buffer = ReplayBuffer(config['n_steps_total'], observation_dim, action_dim)
        task = generate_trajectories(env, 128, config['n_steps_total'])
        replay_buffer.add_trajectory(task)
        replay_buffer.save(
            f'/run/media/rafael/HDD/macaw/{env_name}/buffers_{env_name}_train_{i + start_index}_sub_task_0.hdf5')
        with open(f'/run/media/rafael/HDD/macaw/{env_name}/env_{env_name}_train_task{i + start_index}.pkl', 'wb') as f:
            pkl.dump(task_params, f)
            f.close()
    env.close()


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
        if type(self.model.action_space) is Discrete:
            action = np.zeros(10)
            action[data['actions'][0]] = 1
        else:
            action = data['actions']
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select environment and the amount of tasks to generate.')
    parser.add_argument('--env', type=str, choices=['bandit', 'ant_dir'], required=True)
    parser.add_argument('--num_tasks', type=int, required=True)
    parser.add_argument('--start_index', type=int, required=True, default=0)
    parser.add_argument('--instances', type=int, required=False)
    parser.add_argument('--type', type=int, choices=range(1, 5), required=False)
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=False)
    args = parser.parse_args()
    if args.env == 'ant_dir' and args.type is None:
        parser.error("Argument --env ant_dir requires --type.")
    if args.env == 'bandit' and args.type is not None:
        parser.error("Argument --env bandit does not require --type.")

    if args.mode == 'test':
        np.random.seed(69 + args.start_index)
    elif args.mode == 'train':
        np.random.seed(420 + args.start_index)

    if args.instances is None:
        generate_data(args.env, args.num_tasks, args.start_index, args.type)
    else:
        set_start_method('spawn')
        count = 0
        count_active = 0
        processes = []
        while (count != args.num_tasks):
            if count_active != args.instances:
                subprocess = Process(target=generate_data,
                                     args=(args.env, 1, args.start_index + count, args.type))
                subprocess.start()
                print('Started process: ' + str(count))
                count += 1
                count_active += 1
                processes.append(subprocess)
            for i in range(len(processes)):
                if processes[i].is_alive() == False:
                    processes.pop(i)
                    count_active -= 1
                    break
