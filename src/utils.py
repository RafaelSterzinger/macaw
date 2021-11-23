from typing import NamedTuple, List

import h5py
import numpy as np
import torch
import torch.nn as nn
import os
import random


class RunningEstimator(object):
    def __init__(self):
        self._mu = None
        self._mu2 = None
        self._n = 0

    def mean(self):
        return self._mu

    def var(self):
        return self._mu2 - self._mu ** 2
    
    def std(self):
        return (self.var() + 1e-8) ** 0.5

    def add(self, xs):
        if isinstance(xs, torch.Tensor):
            xs = xs.detach()
        if self._mu is None:
            self._mu = xs.mean()
            self._mu2 = (xs ** 2).mean()
        else:
            self._mu += ((xs - self._mu) * (1 / (self._n + 1))).mean()
            self._mu2 += ((xs**2 - self._mu2) * (1/(self._n+1))).mean()

        self._n += 1


def argmax(module: nn.Module, arg: torch.tensor):
    print('Computing argmax')
    arg.requires_grad = True
    opt = torch.optim.Adam([arg], lr=0.1)
    for idx in range(1000):
        out = module(arg)
        loss = -out
        prev_arg = arg.clone()
        loss.backward()
        opt.step()
        opt.zero_grad()
        module.zero_grad()
        d = (arg-prev_arg).norm(2)
        if d < 1e-4:
            print('breaking')
            break
    return arg, out


def kld(p, q):
    p_mu = p[:,:p.shape[-1] // 2]
    q_mu = q[:,:q.shape[-1] // 2]

    p_std = (p[:,p.shape[-1] // 2:] / 2).exp()
    q_std = (q[:,q.shape[-1] // 2:] / 2).exp()
    dp = torch.distributions.Normal(p_mu, p_std)
    dq = torch.distributions.Normal(q_mu, q_std)

    return torch.distributions.kl_divergence(dp, dq).sum(-1)
    

class Experience(NamedTuple):
    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    reward: float
    done: bool


class ReplayBuffer(object):
    @classmethod
    def from_dict(self, size: int, d: dict, silent: bool):
        print(f'Building replay buffer of size {size}')
        obs_dim = d['obs'].shape[-1]
        action_dim = d['actions'].shape[-1]
        buf = ReplayBuffer(size, obs_dim, action_dim, silent=silent)
        buf._obs[:d['obs'].shape[0]] = d['obs']
        buf._actions[:d['obs'].shape[0]] = d['actions']
        buf._rewards[:d['obs'].shape[0]] = d['rewards']
        buf._mc_rewards[:d['obs'].shape[0]] = d['mc_rewards']
        buf._terminals[:d['obs'].shape[0]] = d['dones']
        buf._terminal_obs[:d['obs'].shape[0]] = d['terminal_obs']
        buf._terminal_discounts[:d['obs'].shape[0]] = d['terminal_discounts']
        buf._next_obs[:d['obs'].shape[0]] = d['next_obs']

        buf._write_location = d['obs'].shape[0]
        buf._stored_steps = d['obs'].shape[0]

        return buf
    
    def __init__(self, size: int, obs_dim: int, action_dim: int, discount_factor: float = 0.99,
                 immutable: bool = False, load_from: str = None, silent: bool = False, skip: int = 1,
                 stream_to_disk: bool = False, mode: str = 'end'):
        if size == -1 and load_from is None:
            print("Can't have size == -1 and no offline buffer - defaulting to 1M steps")
            size = 1000000

        self.immutable = immutable
        self.stream_to_disk = stream_to_disk
        
        if load_from is not None:
            f = h5py.File(load_from, 'r')
            if size == -1:
                size = f['obs'].shape[0]
        
        needs_to_load = True
        size //= skip
        if stream_to_disk:
            name = os.path.splitext(os.path.basename(os.path.normpath(load_from)))[0]
            if os.path.exists('/scr-ssd'):
                path = f'/scr-ssd/em7/{name}'
            else:
                path = f'/scr/em7/{name}'
            if os.path.exists(path):
                if not silent:
                    print(f'Using existing replay buffer memmap at {path}')
                needs_to_load = False
                self._obs = np.memmap(f'{path}/obs.array', mode='r', shape=(size, obs_dim), dtype=np.float32)
                self._actions = np.memmap(f'{path}/actions.array', mode='r', shape=(size, action_dim), dtype=np.float32)
                self._rewards = np.memmap(f'{path}/rewards.array', mode='r', shape=(size, 1), dtype=np.float32)
                self._mc_rewards = np.memmap(f'{path}/mc_rewards.array', mode='r', shape=(size, 1), dtype=np.float32)
                self._terminals = np.memmap(f'{path}/terminals.array', mode='r', shape=(size, 1), dtype=np.bool)
                self._terminal_obs = np.memmap(f'{path}/terminal_obs.array', mode='r', shape=(size, obs_dim), dtype=np.float32)
                self._terminal_discounts = np.memmap(f'{path}/terminal_discounts.array', mode='r', shape=(size, 1), dtype=np.float32)
                self._next_obs = np.memmap(f'{path}/next_obs.array', mode='r', shape=(size, obs_dim), dtype=np.float32)
            else:
                if not silent:
                    print(f'Creating replay buffer memmap at {path}')
                os.makedirs(path)
                self._obs = np.memmap(f'{path}/obs.array', mode='w+', shape=(size, obs_dim), dtype=np.float32)
                self._actions = np.memmap(f'{path}/actions.array', mode='w+', shape=(size, action_dim), dtype=np.float32)
                self._rewards = np.memmap(f'{path}/rewards.array', mode='w+', shape=(size, 1), dtype=np.float32)
                self._mc_rewards = np.memmap(f'{path}/mc_rewards.array', mode='w+', shape=(size, 1), dtype=np.float32)
                self._terminals = np.memmap(f'{path}/terminals.array', mode='w+', shape=(size, 1), dtype=np.bool)
                self._terminal_obs = np.memmap(f'{path}/terminal_obs.array', mode='w+', shape=(size, obs_dim), dtype=np.float32)
                self._terminal_discounts = np.memmap(f'{path}/terminal_discounts.array', mode='w+', shape=(size, 1), dtype=np.float32)
                self._next_obs = np.memmap(f'{path}/next_obs.array', mode='w+', shape=(size, obs_dim), dtype=np.float32)
                self._obs.fill(float('nan'))
                self._actions.fill(float('nan'))
                self._rewards.fill(float('nan'))
                self._mc_rewards.fill(float('nan'))
                self._terminals.fill(float('nan'))
                self._terminal_obs.fill(float('nan'))
                self._terminal_discounts.fill(float('nan'))
                self._next_obs.fill(float('nan'))
        else:
            self._obs = np.full((size, obs_dim), float('nan'), dtype=np.float32)
            self._actions = np.full((size, action_dim), float('nan'), dtype=np.float32)
            self._rewards = np.full((size, 1), float('nan'), dtype=np.float32)
            self._mc_rewards = np.full((size, 1), float('nan'), dtype=np.float32)
            self._terminals = np.full((size, 1), False, dtype=np.bool)
            self._terminal_obs = np.full((size, obs_dim), float('nan'), dtype=np.float32)
            self._terminal_discounts = np.full((size, 1), float('nan'), dtype=np.float32)
            self._next_obs = np.full((size, obs_dim), float('nan'), dtype=np.float32)

        self._size = size
        if load_from is None:
            self._stored_steps = 0
            self._discount_factor = discount_factor
        else:
            if f['obs'].shape[-1] != self.obs_dim:
                raise RuntimeError(f"Loaded data has different obs_dim from new buffer ({f['obs'].shape[-1]}, {self.obs_dim})")
            if f['actions'].shape[-1] != self.action_dim:
                raise RuntimeError(f"Loaded data has different action_dim from new buffer ({f['actions'].shape[-1]}, {self.action_dim})")

            stored = f['obs'].shape[0]
            n_seed = min(stored, self._size * skip)
            self._stored_steps = n_seed // skip

            if needs_to_load:
                if not silent:
                    print(f'Loading trajectories from {load_from}')
                if stored > self._size * skip:
                    if not silent:
                        print(f"Attempted to load {stored} offline steps into buffer of size {self._size}.")
                        print(f"Loading only the **{mode}** {n_seed//skip} steps from offline buffer")

                chunk_size = n_seed
                self._discount_factor = f['discount_factor'][()]
                if mode == 'end':
                    h5slice = slice(-chunk_size, stored)
                elif mode == 'middle':
                    center = stored // 2
                    h5slice = slice(center // 2 - chunk_size // 2,center // 2 + chunk_size // 2)
                elif mode == 'start':
                    h5slice = slice(chunk_size)
                else:
                    raise Exception(f'No such mode {mode}')

                self._obs[:self._stored_steps] = f['obs'][h5slice][::skip]
                self._actions[:self._stored_steps] = f['actions'][h5slice][::skip]
                self._rewards[:self._stored_steps] = f['rewards'][h5slice][::skip]
                self._mc_rewards[:self._stored_steps] = f['mc_rewards'][h5slice][::skip]
                self._terminals[:self._stored_steps] = f['terminals'][h5slice][::skip]
                self._terminal_obs[:self._stored_steps] = f['terminal_obs'][h5slice][::skip]
                self._terminal_discounts[:self._stored_steps] = f['terminal_discounts'][h5slice][::skip]
                self._next_obs[:self._stored_steps] = f['next_obs'][h5slice][::skip]

            f.close()

        self._write_location = self._stored_steps % self._size

    @property
    def obs_dim(self):
        return self._obs.shape[-1]

    @property
    def action_dim(self):
        return self._actions.shape[-1]

    def __len__(self):
        return self._stored_steps

    def save(self, location: str):
        f = h5py.File(location, 'w')
        f.create_dataset('obs', data=self._obs[:self._stored_steps], compression='lzf')
        f.create_dataset('actions', data=self._actions[:self._stored_steps], compression='lzf')
        f.create_dataset('rewards', data=self._rewards[:self._stored_steps], compression='lzf')
        f.create_dataset('mc_rewards', data=self._mc_rewards[:self._stored_steps], compression='lzf')
        f.create_dataset('terminals', data=self._terminals[:self._stored_steps], compression='lzf')
        f.create_dataset('terminal_obs', data=self._terminal_obs[:self._stored_steps], compression='lzf')
        f.create_dataset('terminal_discounts', data=self._terminal_discounts[:self._stored_steps], compression='lzf')
        f.create_dataset('next_obs', data=self._next_obs[:self._stored_steps], compression='lzf')
        f.create_dataset('discount_factor', data=self._discount_factor)
        f.close()
    
    def add_trajectory(self, trajectory: List[Experience], force: bool = False):
        if self.immutable and not force:
            raise ValueError('Cannot add trajectory to immutable replay buffer')

        mc_reward = 0
        terminal_obs = None
        terminal_factor = 1
        for idx, experience in enumerate(trajectory[::-1]):
            if terminal_obs is None:
                terminal_obs = experience.next_state

            self._obs[self._write_location] = experience.state
            self._next_obs[self._write_location] = experience.next_state
            self._actions[self._write_location] = experience.action
            self._rewards[self._write_location] = experience.reward
            self._terminals[self._write_location] = experience.done
            self._terminal_obs[self._write_location] = terminal_obs

            terminal_factor *= self._discount_factor
            self._terminal_discounts[self._write_location] = terminal_factor

            mc_reward = experience.reward + self._discount_factor * mc_reward
            self._mc_rewards[self._write_location] = mc_reward

            self._write_location += 1
            self._write_location = self._write_location % self._size
            
            if self._stored_steps < self._size:
                self._stored_steps += 1


    def add_trajectories(self, trajectories: List[List[Experience]], force: bool = False):
        for trajectory in trajectories:
            self.add_trajectory(trajectory, force)

    def sample(self, batch_size, return_dict: bool = False, return_both: bool = False,
               noise: bool = False, contiguous: bool = False):
        if contiguous:
            idx = np.random.randint(0, self._stored_steps - batch_size)
            idxs = slice(idx, idx + batch_size)
        else:
            idxs = np.array(random.sample(range(self._stored_steps), batch_size))

        obs = self._obs[idxs]
        actions = self._actions[idxs]
        next_obs = self._next_obs[idxs]
        terminal_obs = self._terminal_obs[idxs]
        terminal_discounts = self._terminal_discounts[idxs]
        dones = self._terminals[idxs]
        rewards = self._rewards[idxs]
        mc_rewards = self._mc_rewards[idxs]
        
        batch = np.concatenate((obs, actions, next_obs, terminal_obs, terminal_discounts, dones, rewards, mc_rewards), 1)
        if noise:
            std = batch.std(0) * np.sqrt(batch_size)
            mu = np.zeros(std.shape)
            noise = np.random.normal(mu, std, batch.shape).astype(np.float32)
            batch = batch + noise
        batch_dict =  {
            'obs': obs,
            'actions': actions,
            'next_obs': next_obs,
            'terminal_obs': terminal_obs,
            'terminal_discounts': terminal_discounts,
            'dones': dones,
            'rewards': rewards,
            'mc_rewards': mc_rewards
        }
        if return_both:
            return batch, batch_dict
        elif not return_dict:
            return batch
        else:
            return batch_dict