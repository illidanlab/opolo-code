import argparse
import time
import difflib
import os
from collections import OrderedDict
from pprint import pprint
import warnings
import importlib
from settings import CONFIGS
import tensorflow as tf
# For pybullet envs
warnings.filterwarnings("ignore")
import gym
import mujoco_py
try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None
import numpy as np
import yaml
try:
    import highway_env
except ImportError:
    highway_env = None

from mpi4py import MPI
from stable_baselines.common import set_global_seeds

from stable_baselines.ppo2.ppo2 import constfn
from stable_baselines.results_plotter import load_results, ts2xy

from utils import create_env, create_test_env, ALGOS, linear_schedule, get_latest_run_id, get_wrapper_class
from utils.noise import LinearNormalActionNoise
from stable_baselines.gail import generate_expert_traj_mujoco, generate_expert_traj
from stable_baselines.gail.dataset.dataset import ExpertDataset


best_mean_reward, n_steps = -np.inf, 0
PATH_PREFIX = '..'
def is_mujoco(env):
    for name in ['Hopper', 'Half', 'Walker', 'Reacher', 'Ant', 'Humanoid', 'Pendulum', 'Pusher', 'Swimmer', 'Thrower', 'Striker']:
        if name in env:
            return True
    return False

def check_if_atari(env_id):
    is_atari = False
    no_skip = False
    for game in ['Gravitar', 'MontezumaRevenge', 'Pitfall', 'Qbert', 'Pong']:
        if game in env_id:
            is_atari = True
            if 'NoFrameskip' in env_id:
                no_skip = True
            return is_atari, no_skip


def need_demo(config):
    shaping_mode = config['shaping_mode']
    if 'gail' in shaping_mode:
        return True
    if 'gaifo' in shaping_mode:
        return True
    if 'gaifso' in shaping_mode:
        return True
    if 'bco' in shaping_mode:
        return True

    return False

def a2c_callback(log_dir, mode, max_score=None):
    def callback(_locals, _globals):
        """
        Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
        :param _locals: (dict)
        :param _globals: (dict)
        """
        global n_steps, best_mean_reward
        # Print stats every 20 calls
        if (n_steps + 1) % 500 == 0:
            # Evaluate policy training performance
            x, y = ts2xy(load_results(log_dir), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                print(x[-1], 'timesteps')
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

                # New best model, you could save the agent here
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    # Example for saving best model
                    print("Saving new best model")
                    _locals['self'].save(os.path.join(log_dir, 'best_model.pkl'))
                if 'train' in mode and max_score is not None and mean_reward > max_score:
                    print("Stop training.")
                    return False
        n_steps += 1
        return True
    return callback


def eval_mujoco_model(args, model, env, step=50000):
    ###########################
    # Force deterministic for DQN, DDPG, SAC and HER (that is a wrapper around)
    deterministic = args.algo in ['dqn', 'ddpg', 'her', "sail", "opolo", "bcq", "dac", 'dqnrrs'] or 'td3' in args.algo
    #deterministic = False
    episode_reward = 0.0
    episode_rewards = []
    ep_len = 0
    # For HER, monitor success rate
    obs = env.reset()[0]
    successes = []
    from stable_baselines.common.math_util import unscale_action


    for i in range(step):
        #action, _ = model.predict(obs, deterministic=deterministic)
        action, _ = model.predict(obs, deterministic=False)


        #if args.algo == 'sac':
        #    action = unscale_action(args.env.action_space, action)
        # Clip Action to avoid out of bound errors
        if isinstance(env.action_space, gym.spaces.Box):
            action = np.clip(action, -1, 1)
        obs, reward, done, infos = env.step(action)
        if not args.no_render:
            env.render('human')

        ep_len += 1

        n_envs = hyperparams.get('n_envs', 1)
        if n_envs == 1:
            if done and not is_atari and args.verbose > 0:
                x, y = ts2xy(load_results(args.log_dir), 'timesteps')
                if len(x) > 0:
                    mean_reward = np.mean(y[-100:])
                    episode_reward = y[-1]
                    print(x[-1], 'timesteps')
                    print("Length: {}, Mean reward: {:.2f} - Last episode reward: {:.2f}".format(ep_len, mean_reward, episode_reward))
                episode_rewards.append(episode_reward)
                ep_len = 0


    if args.verbose > 0 and len(episode_rewards) > 0:
        print("Mean reward: {:.2f}".format(np.mean(episode_rewards)))


def eval_atari_model(args, model, env, step=30000):
    ###########################
    # Force deterministic for DQN, DDPG, SAC and HER (that is a wrapper around)
    deterministic = args.algo in ['dqn', 'ddpg', 'sac', 'her', "sail", "opolo", "bcq", "dac", 'dqnrrs'] or 'td3' in args.algo
    episode_reward = 0.0
    episode_rewards = []
    # For HER, monitor success rate
    obs = env.reset()


    for i in range(step):
        action, _ = model.predict(obs, deterministic=deterministic)
        action_prob = model.action_probability(obs)
        #if i % 1000 == 0:
        #    print(action_prob)
        # Clip Action to avoid out of bound errors
        if isinstance(env.action_space, gym.spaces.Box):
            action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, done, infos = env.step(action)
        if not args.no_render:
            env.render('human')


        n_envs = hyperparams.get('n_envs', 1)
        if n_envs == 1:
            if infos is not None:
                episode_infos = infos[0].get('episode')
                if episode_infos is not None:
                    episode_reward = episode_infos['r']
                    print("Atari Episode Score: {:.2f}".format(episode_reward))
                    print("Atari Episode Length", episode_infos['l'])
                    episode_rewards.append(episode_reward)

    if args.verbose > 0 and len(episode_rewards) > 0:
        print("Mean reward: {:.2f}, Variance: {:.2f}".format(np.mean(episode_rewards), np.std(episode_rewards)))

    # Workaround for https://github.com/openai/gym/issues/893
    if not args.no_render:
        if args.n_envs == 1 and 'Bullet' not in env_id and isinstance(env, VecEnv):
            # DummyVecEnv
            # Unwrap env
            while isinstance(env, VecNormalize) or isinstance(env, VecFrameStack):
                env = env.venv
            env.envs[0].env.close()
        else:
            # SubprocVecEnv
            env.close()


def load_expert_hyperparams(args):
    with open('../hyperparams/{}.yml'.format(args.algo), 'r') as f:
        hyperparams_dict = yaml.load(f)
        if env_id in list(hyperparams_dict.keys()):
            hyperparams = hyperparams_dict[env_id]
        elif is_atari:
            hyperparams = hyperparams_dict['atari']
        else:
            raise ValueError("Hyperparameters not found for {}-{}".format(args.algo, env_id))

    # Sort hyperparams that will be saved
    saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])

    algo_ = args.algo
    if args.verbose > 0:
        pprint(saved_hyperparams)
    return hyperparams, saved_hyperparams


def initiate_hyperparams(args, hyperparams):
    n_envs = hyperparams.get('n_envs', 1)

    if args.verbose > 0:
        print("Using {} environments".format(n_envs))

    # Create learning rate schedules for ppo2 and sac
    if args.algo in ["ppo2", "sac", "dac", "sail", "sacil"] or 'td3' in args.algo:
        for key in ['learning_rate', 'cliprange', 'cliprange_vf']:
            if key not in hyperparams:
                continue
            if isinstance(hyperparams[key], str):
                schedule, initial_value = hyperparams[key].split('_')
                initial_value = float(initial_value)
                hyperparams[key] = linear_schedule(initial_value)
            elif isinstance(hyperparams[key], (float, int)):
                # Negative value: ignore (ex: for clipping)
                if hyperparams[key] < 0:
                    continue
                hyperparams[key] = constfn(float(hyperparams[key]))
            else:
                raise ValueError('Invalid value for {}: {}'.format(key, hyperparams[key]))
    return hyperparams


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="PongNoFrameskip-v4", help='environment ID')
    parser.add_argument('-tb', '--tensorboard-log', help='Tensorboard log dir', default='tb_logs', type=str)
    parser.add_argument('-i', '--trained-agent', help='Path to a pretrained agent to continue training',
                        default='', type=str)
    parser.add_argument('--algo', help='RL Algorithm', default='dqnrrs',
                        type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument('-n', '--n-timesteps', help='Overwrite the number of timesteps', default=-1,
                        type=int)
    parser.add_argument('--log-interval', help='Override log interval (default: -1, no change)', default=1000,
                        type=int)
    parser.add_argument('-f', '--log-folder', help='Log folder', type=str, default='logs')
    parser.add_argument('--seed', help='Random generator seed', type=int, default=1)
    parser.add_argument('--log-dir', help='Log directory', type=str, default='/tmp/logs') # required=True,
    parser.add_argument('-optimize', '--optimize-hyperparameters', action='store_true', default=False,
                        help='Run hyperparameters search')
    parser.add_argument('--n-jobs', help='Number of parallel threads when optimizing hyperparameters', type=int, default=6)
    parser.add_argument('--n-episodes', help='Number of expert episodes', type=int, default=1)
    parser.add_argument('--no-render', help='If render', default=True)
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1,
                        type=int)
    parser.add_argument('--gym-packages', type=str, nargs='+', default=[], help='Additional external Gym environemnt package modules to import (e.g. gym_minigrid)')
    parser.add_argument(
        '--task',
        type=str,
        default='train')
    args = parser.parse_args()

    # extend log directory with experiment details
    new_log_dir = os.path.join(args.log_dir, args.task, args.algo, args.env, 'rank{}'.format(args.seed))
    args.log_dir = new_log_dir

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    registered_envs = set(gym.envs.registry.env_specs.keys())

    # If the environment is not found, suggest the closest match
    if args.env not in registered_envs:
        try:
            closest_match = difflib.get_close_matches(args.env, registered_envs, n=1)[0]
        except IndexError:
            closest_match = "'no close match found...'"
        raise ValueError('{} not found in gym registry, you maybe meant {}?'.format(env_id, closest_match))

    set_global_seeds(args.seed)

    if args.trained_agent != "":
        #args.trained_agent = os.path.join(args.trained_agent, args.algo, "{}.pkl".format(args.env))
        print(args.trained_agent)
        assert args.trained_agent.endswith('.pkl') and os.path.isfile(args.trained_agent), \
            "The trained_agent must be a valid path to a .pkl file"

    rank = 0
    if MPI.COMM_WORLD.Get_size() > 1:
        print("Using MPI for multiprocessing with {} workers".format(MPI.COMM_WORLD.Get_size()))
        rank = MPI.COMM_WORLD.Get_rank()
        print("Worker rank: {}".format(rank))

        args.seed += rank
        if rank != 0:
            args.verbose = 0
            args.tensorboard_log = ''

    tensorboard_log = os.path.join(args.log_dir, 'tb')

    is_atari = False
    if 'NoFrameskip' in env_id:
        is_atari = True

    print("=" * 10, env_id, "=" * 10)

    #############################################
    # Load hyperparameters & create environment
    #############################################
    hyperparams, saved_hyperparams = load_expert_hyperparams(args)
    # replace str in hyperparams to be real entities
    hyperparams = initiate_hyperparams(args, hyperparams)
    # Should we overwrite the number of timesteps?
    if args.n_timesteps > 0:
        if args.verbose:
            print("Overwriting n_timesteps with n={}".format(args.n_timesteps))
        n_timesteps = args.n_timesteps
    else:
        n_timesteps = int(hyperparams['n_timesteps'] * 1.0 )
    print(n_timesteps)
    #exit()
        #n_timesteps = int(hyperparams['n_timesteps'] )

    config = CONFIGS[args.env]
    config['n_episodes'] = args.n_episodes
    config['n_jobs'] = args.n_jobs
    config['shaping_mode'] = args.task
    config['use_idm'] = 'idm' in args.task # use inverse model to infer teacher action for BC
    config['use_prior'] = 'prior' in args.task
    config['use_hindsight'] = 'hindsight' in args.task # use ground-truth teacher action for BC
    config['sparse'] = False

    data_save_path = '../expert_logs'
    max_score = config['optimal_score']
    data_save_dir = os.path.join(data_save_path, "expert_data_no_img_{}_scores_{}_episodes_{}.npz".format(args.env.split('-')[0], max_score, config['n_episodes']))
    config['expert_data_path'] = data_save_dir

    #args.log_dir = args.log_dir.replace('eval-bc', 'eval-bc-episode-{}'.format(config['n_episodes']))
    os.makedirs(args.log_dir, exist_ok=True)
    env, hyperparams, normalize = create_env(args, hyperparams, is_atari, n_timesteps)

    #n_timesteps = int(1e7)
    ######################################
    # build a new model or a trained agent
    ######################################

    policy = hyperparams['policy']
    del hyperparams['policy']

    #####################################
    # build model
    #####################################

    # Train an agent from scratch
    policy = 'MlpPolicy'
    policy_kwargs = dict(act_fun=tf.nn.tanh, layers=[64, 64, 64])
    kwargs = {}
    if args.log_interval > -1:
        kwargs = {'log_interval': args.log_interval}
    if 'test' in args.task:
        test_env = create_test_env(args, hyperparams)
    else:
        test_env = None
    if 'env_wrapper' in hyperparams.keys():
        del hyperparams['env_wrapper']

    config['log_dir'] = args.log_dir
    config['test_env'] = test_env
    print(config)
    time.sleep(3)

    ##############################################################
    # begin learning, saving best model through each call back
    ##############################################################
    cb_func = a2c_callback(args.log_dir, args.task)
    if 'eval' in args.task:
        model_file = os.path.join("{}/{}".format(args.log_dir.replace('eval-',''), "best_model.pkl"))
        assert os.path.isfile(model_file), \
            " The eval_model_path must be a valid path to a pkl or zip file, but no file found at {} ".format(model_file)
        model = ALGOS[args.algo].load(
            model_file,
            env=env,
            config=config,
            verbose=args.verbose)
        if is_mujoco(args.env):
            eval_mujoco_model(args, model, env)
    else:
        if need_demo(config):
            print(data_save_dir)
            assert os.path.isfile(data_save_dir)
            print("Loading Demo Data: {}".format(data_save_dir))
        model = ALGOS[args.algo](
            policy,
            env=env,
            tensorboard_log=tensorboard_log,
            verbose=args.verbose,
            config=config,
            **hyperparams)
        model.learn(
            n_timesteps,
            callback=cb_func,
            **kwargs)
        with open(os.path.join(args.log_dir, 'config.yml'), 'w') as f:
            yaml.dump(saved_hyperparams, f)











