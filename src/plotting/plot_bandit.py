import os
import pickle as pkl

import numpy as np
import matplotlib.pyplot as plt

LINES = ['-', '--', '-.', ':']

if __name__ == "__main__":
    path = '../../log'
    for dir in os.listdir(path):
        type = int(dir[-1])
        label = str(type) + ' tasks'
        dir = os.path.join(path, dir)
        means = []
        for subdir in os.listdir(dir):
            subdir = os.path.join(dir, subdir)
            with open(os.path.join(subdir, 'mean.pkl'), 'rb') as f:
                df = pkl.load(f)
                f.close()
            means.append(df['value'].to_numpy())
        means = np.asarray(list(map(lambda x: x[:20], means)))
        mean = np.mean(means, axis=0)
        std = np.std(means, axis=0)
        ticks = np.arange(20) * 250
        ci = 1.96 * std / np.sqrt(mean.shape[0])

        plt.plot(ticks, mean, label=label)
        plt.fill_between(ticks, (mean - ci), (mean + ci), alpha=.2)
    plt.legend()
    plt.title('Bernoulli-Bandit Environment')
    plt.xlabel('Training Iterations')
    plt.ylabel('Average Reward')
    #plt.xscale('log')
    plt.grid(True)
    plt.savefig('bandit.pdf')
