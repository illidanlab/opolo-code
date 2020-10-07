import seaborn as sns
import argparse
import matplotlib.pyplot as plt
import collections
import pandas as pd
import numpy as np
import os
from settings import CONFIGS

legend_loc = "lower right"
def plot_results_game(experimental_data, title, num_iter, args, demo_score=None, demo_std=None):
    fig, ax = plt.subplots(figsize=(15,10))
    g = sns.tsplot(data=experimental_data, time='iteration', unit='run_number', condition='agent', value='train_episode_reward', ax=ax, ci=50)
    if demo_score is not None:
        x = [i * 10 for i in range(num_iter // 10)]
        y = [demo_score for _ in x]
        e = [demo_std for _ in x]
        ax.errorbar(x, y, e, label='Expert', marker='*',color='grey')

    fontsize = 40
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')

    fontsize = "45"
    title_fontsize = 48
    yaxis_label, xaxis_label = "Returns", "Interaction Steps ({}e{})".format(str(args.steps).strip('0'),int(np.log10(args.steps)))
    title_axis_font = {'size': title_fontsize, 'weight': 'bold'}
    xylabel_axis_font = {'size': fontsize, 'weight': 'bold'}
    ax.set_ylabel(yaxis_label, **xylabel_axis_font)
    ax.set_xlabel(xaxis_label, **xylabel_axis_font)
    ax.set_title(title, **title_axis_font)
    legend_properties = {'weight':'bold','size':"35"}
    if args.plot:
        ax.legend(loc=legend_loc, prop=legend_properties,ncol=1)#,bbox_to_anchor=(-0.1,0.5))
    else:
        g.legend_.remove()

    plt.tight_layout()
    # plt.show()
    if args.savefig:
        title = title.replace(" ", "")
        title = '{}-ep{}'.format(title, args.episodes)
        if args.ab:
            title = '{}-ab{}'.format(title, args.ab)
        if args.title != '':
            title = '{}-{}'.format(title, args.title)
        plt.savefig('../figs/{}.pdf'.format(title))
        plt.close()
    else:
        plt.show()
    os.system("mkdir -p ../figs")
    print("Figure {}.pdf saved to {}".format(title, "../figs/"))

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def read_log(game, algo, agent_name, log_path, num_steps_per_iteration, num_iter,seeds, warmup=0):
    results = []
    monitor_file = 'agent0.monitor.csv' if 'ddpg' not in algo else 'monitor.csv'
    for seed in seeds:
        csv_file = os.path.join(log_path, 'rank{}'.format(seed), monitor_file)
        df = pd.read_csv(csv_file, header=1).dropna()
        df.loc[:, 'l'] = df.loc[:, 'l'].cumsum()
        eplen = np.array(df.loc[:, 'l'].tolist())
        eprewmean = np.array(df.loc[:, 'r'].tolist())
        # construct arary
        for i in range(num_iter):
            results_per_iter = []
            step_begin = int(i * num_steps_per_iteration + warmup)
            step_end = int((i+1) * num_steps_per_iteration + warmup)
            eprew_iter = eprewmean[(eplen>step_begin) & (eplen <=step_end)]
            results_per_iter.append(agent_name)
            results_per_iter.append(game)
            results_per_iter.append(i)  #iteration
            results_per_iter.append(np.mean(eprew_iter))  # train episode reward
            results_per_iter.append(seed)  # run number
            results.append(results_per_iter)

    experimental_data = pd.DataFrame.from_records(results,  columns=["agent", "game", "iteration", "train_episode_reward", "run_number"])
    return experimental_data

def get_stepwise_results(legend, algo, log_path, num_iter, chunk_size, window_size, seeds):
    # get peformance avergage for the first 1e6 steps
    # starting index: 0.1 M , 0.3 M, ..., 0.9M
    # ending index: 0.2 M, 0.4 M, ..., 1 M
    results = []
    monitor_file = 'agent0.monitor.csv' if 'ddpg' not in algo else 'monitor.csv'
    results = []
    for seed in seeds:
        csv_file = os.path.join(log_path, 'rank{}'.format(seed), monitor_file)
        df = pd.read_csv(csv_file, header=1).dropna()
        df.loc[:, 'l'] = df.loc[:, 'l'].cumsum()
        eplen = np.array(df.loc[:, 'l'].tolist())
        eprewmean = np.array(df.loc[:, 'r'].tolist())
        #print(eplen[-1])
        # construct arary
        results_per_seed= []
        for i in range(num_iter):
            step_begin = int(i * chunk_size )
            step_end = int(i * chunk_size + window_size)
            eprew_iter = eprewmean[(eplen>step_begin) & (eplen <=step_end)]
            results_per_seed.append(np.mean(eprew_iter))  # train episode reward
        results.append(results_per_seed)
    result = np.mean(np.array(results), axis=0)
    result = [r.round(2) for r in result]
    print("\n{}, {}\n".format(legend,result))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="HalfCheetah-v2", help='environment ID')
    parser.add_argument('--algo', type=str, default="sail", help='Algorithm')
    parser.add_argument('--shift', type=int, default=0, help='seed index shift')
    parser.add_argument('--seeds', type=int, default=5, help='number of seeds')
    parser.add_argument('--episodes', type=int, default=1, help='number of demonstration episodes')
    parser.add_argument('--warmup', type=int, default=0, help='number of warmup samples')
    parser.add_argument('--timesteps', type=int, default=int(1e6), help='Time steps')
    parser.add_argument('--steps', type=int, default=5000, help='num steps per iteration')
    parser.add_argument('--rewards', type=str, choices=['dense', 'sparse'], default='dense', help='Environment Rewards')
    parser.add_argument('--quality', type=str, choices=['near', 'sub'], default='sub', help='Environment Rewards')
    parser.add_argument('--legends', type=str, default='', help="Algorithms to be ploted, split by comma")
    parser.add_argument('--plot', type=int, default=0, help="Plot legend (1) or not (0)")
    parser.add_argument('--ab', type=int, default=0, help="Figures for Ablation Study")
    parser.add_argument('--title', type=str, default='', help="Title postfix")
    parser.add_argument('--savefig', type=bool, default=True, help="Save Fig")
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

legends = [l.strip() for l in args.legends.split(',')]
new_legends = []
for legend in legends:
    new_legend = legend
    task = legend
    algo_prefix, algo_postfix = '', ''
    if 'trpo' in legend:
        algo = algo_prefix = 'trpo'
    elif 'ddpgfd' in legend:
        algo = algo_prefix = 'ddpgfd'
    elif 'ddpg' in legend:
        algo = algo_prefix = 'ddpg'
    elif 'sacil' in legend:
        algo = algo_prefix = 'sacil'
    elif 'sacfo' in legend:
        algo = algo_prefix = 'sacfo'
    elif 'sac' in legend:
        algo = algo_prefix = 'sac'
    else:
        algo_prefix = 'td3'

    if 'gaifo' in legend:
        algo_postfix = 'gaifo'
        new_legend = 'GAIfO'
    elif 'gaifso' in legend:
        algo_postfix = 'gaifso'
    elif 'gail' in legend:
        algo_postfix = 'gail'
        new_legend = 'GAIL'
    elif 'bco' in legend:
        algo_postfix = 'bco'
        new_legend = 'BCO'
    elif 'lfo' in legend:
        algo_postfix = 'lfo'
    elif 'dacfo' in legend:
        algo_postfix = 'dacfo'
        new_legend = 'DACfO'
    elif 'dac' in legend:
        algo_postfix = 'dac'
        new_legend = 'DAC'
    if algo_postfix:
        algo = algo_prefix + algo_postfix
    legend = new_legend
    path = os.path.join(path_prefix, task, algo, env)
    experimental_data = read_log(env, algo, legend, path, args.steps, num_iter, seeds, warmup=args.warmup)
    experimental_datas.append(experimental_data)


figure_name = env
all_agents_data = pd.concat(experimental_datas, axis=0)
savefig = True
#### get Demonstration line #####
data_save_path = '{}/expert_logs'.format(path_prefix)
max_score = CONFIGS[args.env]['optimal_score']
expert_path = os.path.join(data_save_path, "expert_data_no_img_{}_scores_{}_episodes_{}.npz".format(args.env.split('-')[0], max_score, args.episodes))

if os.path.isfile(expert_path):
    traj_data = np.load(expert_path, allow_pickle=True)
    demo_score = np.mean(traj_data['episode_returns'])
    demo_std = np.std(traj_data['episode_returns'])
    #print("Demonstration Average: {}, Standard Deviation: {}".format(demo_score, demo_std))
else:
    demo_score, demo_std = None, None

plot_results_game(all_agents_data, figure_name, num_iter, args, demo_score=demo_score, demo_std=demo_std)
