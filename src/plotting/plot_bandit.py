import os
import pickle as pkl
import re

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    path = '../../log'
    stds = []
    means = []
    cis = []
    types = []
    for dir in os.listdir(path):
        dir = os.path.join(path, dir)
        type = int(re.findall('\d+', dir)[0])
        types.append(type)
        mean = []
        for subdir in os.listdir(dir):
            subdir = os.path.join(dir, subdir)
            with open(os.path.join(subdir, 'mean.pkl'), 'rb') as f:
                df = pkl.load(f)
                f.close()
            mean.append(df['value'].to_numpy())
        mean = np.asarray(list(map(lambda x: x[3000 // 250:], mean)))
        size = mean.shape[0] * mean.shape[1]
        std = np.std(mean)
        ci = 1.96 * std / np.sqrt(size)
        mean = np.mean(mean)
        means.append(np.mean(mean))
        stds.append(std)
        cis.append(ci)
    sort = np.argsort(types)
    types = np.take_along_axis(np.asarray(types), sort, axis=0)
    means = np.take_along_axis(np.asarray(means), sort, axis=0)
    stds = np.take_along_axis(np.asarray(stds), sort, axis=0)
    cis = np.take_along_axis(np.asarray(cis), sort, axis=0)
    ticks = list(range(5))

    plt.plot(ticks, means, 's', color='#1f77b4', label='MACAW')
    plt.plot(ticks, means, color='#1f77b4')
    plt.fill_between(ticks, (means - cis), (means + cis), alpha=.2)
    opt = 894
    plt.plot(ticks, [opt] * 5, '--', color='k', label='Expected Optimal Reward')
    plt.legend()
    plt.title('Bernoulli-Bandit Environment')
    plt.xlabel('Number of Training Tasks')
    plt.ylabel('Asymptotic Reward')
    plt.xticks(ticks, types)
    plt.grid(True)
    plt.savefig('bandit.pdf')
