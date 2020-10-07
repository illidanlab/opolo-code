import argparse
import pandas as pd
import numpy as np
import os
from settings import CONFIGS

legend_loc = "upper left"

def eval_avg_results(path, args):
    means, stds = [], []
    seeds=[i + args.shift for i in range(1, args.seeds + 1)] 
    for seed in seeds:
        file_name = '{}/rank{}/agent0.monitor.csv'.format(path, seed)
        bc_mean, bc_std = get_score(file_name)
        means.append(bc_mean)
        stds.append(bc_std)
    print('{:.2f} + {:.2f}'.format(np.mean(means), np.mean(stds)))
    print()


def get_score(csv_file):
    df = pd.read_csv(csv_file, header=1).dropna()
    df.loc[:, 'l'] = df.loc[:, 'l'].cumsum()
    eprewmean = np.array(df.loc[:, 'r'].tolist())
    mean_rew, std_rew = np.mean(eprewmean), np.std(eprewmean)
    return mean_rew, std_rew



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="HalfCheetah-v2", help='environment ID')
    parser.add_argument('--algo', type=str, default="sail", help='Algorithm')
    parser.add_argument('--shift', type=int, default=0, help='seee index shift')
    parser.add_argument('--seeds', type=int, default=5, help='number of seeds')
    parser.add_argument('--episodes', type=int, default=1, help='number of demonstration episodes')
    parser.add_argument('--timesteps', type=int, default=int(1e6), help='Time steps')
    parser.add_argument('--steps', type=int, default=10000, help='num steps per iteration')
    parser.add_argument('--legends', type=str, default='ep1-gail-lfd-adaptive-dynamic', help="Methods to plot, split by comma")
    parser.add_argument('--plot', type=int, default=1, help="Plot legend (1) or not (0)")
    parser.add_argument('--test', type=int, default=0, help="Plot Test mode ")
    parser.add_argument('--title', type=str, default='sail', help="Figure Name")
    parser.add_argument('--path-prefix', type=str, help="path to find performance logs", required=True)
    args = parser.parse_args()


experimental_datas = []

timesteps = args.timesteps
path_prefix = args.path_prefix
num_steps_per_iteration = args.steps
num_iter = int(timesteps/args.steps)

algo = args.algo
env=args.env
seeds=[i + args.shift for i in range(1, args.seeds + 1)] # dense, near-optimal

config = CONFIGS[args.env]
#### get Demonstration line #####
data_save_path = '{}/expert_logs'.format(path_prefix)
data_save_dir = os.path.join(data_save_path, "expert_data_no_img_{}_scores_{}_episodes_{}".format(args.env.split('-')[0], config['optimal_score'], args.episodes))
expert_path = '{}.npz'.format(data_save_dir)
#print("Demo Data save in : {}".format(expert_path))
traj_data = np.load(expert_path, allow_pickle=True)
demo_score = np.mean(traj_data['episode_returns'])
demo_std = np.std(traj_data['episode_returns'])
print('Demonstration mean = {:.2f}, std = {:.2f}, episode-num = {}'.format(demo_score, demo_std, args.episodes))

legends = [l.strip() for l in args.legends.split(',')]
for legend in legends:
    task = legend
    algo = args.algo
    algo_prefix, algo_postfix = '', ''
    if 'trpo' in legend:
        algo = algo_prefix = 'trpo'
    else:
        algo_prefix = 'td3'

    if 'gaifo' in legend:
        algo_postfix = 'gaifo'
    elif 'gail' in legend:
        algo_postfix = 'gail'
    elif 'bco' in legend:
        algo_postfix = 'bco'
    #elif 'dicefo' in legend:
    #    algo_postfix = 'dicefo'
    elif 'dacfo' in legend:
        algo_postfix = 'dacfo'
    elif 'dac' in legend:
        algo_postfix = 'dac'
    if algo_postfix:
        algo = algo_prefix + algo_postfix
    path = os.path.join(path_prefix, task, algo, env)
    print(path)
    eval_avg_results(path, args)
