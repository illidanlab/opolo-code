import seaborn as sns
import argparse
import matplotlib.pyplot as plt
import collections
import pandas as pd
import numpy as np
import os
from settings import CONFIGS

from settings import PATH_PREFIX as path_prefix
legend_loc = "lower right"
legend_loc = "upper left"


def eval_avg_results(path, args):
    #best_mean, best_std = -float('inf'), 0
    means, stds = [], []
    seeds=[i + args.shift for i in range(1, args.seeds + 1)] 
    for seed in seeds:
        file_name = '{}/rank{}/final.npy'.format(path, seed)
        data = np.load(file_name)
        avg, std = np.mean(data), np.std(data)
        means.append(avg)
        stds.append(std)
    print('{:.2f} + {:.2f}'.format(np.mean(means), np.mean(stds)))


def get_bc_score(csv_file):
    df = pd.read_csv(csv_file, header=1).dropna()
    df.loc[:, 'l'] = df.loc[:, 'l'].cumsum()
    #eplen = np.array(df.loc[:, 'l'].tolist())
    eprewmean = np.array(df.loc[:, 'r'].tolist())
    mean_rew, std_rew = np.mean(eprewmean), np.std(eprewmean)
    return mean_rew, std_rew



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="HalfCheetah-v2", help='environment ID')
    parser.add_argument('--algo', type=str, default="valuedice", help='Algorithm')
    parser.add_argument('--shift', type=int, default=0, help='seee index shift')
    parser.add_argument('--seeds', type=int, default=3, help='number of seeds')
    parser.add_argument('--episodes', type=int, default=4, help='number of demonstration episodes')
    parser.add_argument('--timesteps', type=int, default=int(1e6), help='Time steps')
    parser.add_argument('--legends', type=str, default='ep1-gail-lfd-adaptive-dynamic', help="Methods to plot, split by comma")
    args = parser.parse_args()



env=args.env
seeds=[i + args.shift for i in range(1, args.seeds + 1)] # dense, near-optimal

config = CONFIGS[args.env]
#### get Demonstration line #####
data_save_path = '{}/expert_logs'.format(path_prefix)
data_save_dir = os.path.join(data_save_path, "expert_data_no_img_{}_scores_{}_episodes_{}".format(args.env.split('-')[0], config['suboptimal_score'], args.episodes))
expert_path = '{}.npz'.format(data_save_dir)
#print("Demo Data save in : {}".format(expert_path))
traj_data = np.load(expert_path, allow_pickle=True)
demo_score = np.mean(traj_data['episode_returns'])
demo_std = np.std(traj_data['episode_returns'])
print('Demonstration mean = {:.2f}, std = {:.2f}, episode-num = {}'.format(demo_score, demo_std, args.episodes))

legends = [l.strip() for l in args.legends.split(',')]
for legend in legends:
    task = legend
    algo = 'valuedice'
    path = os.path.join(path_prefix, task, algo, env)
    print(path)
    eval_avg_results(path, args)
