import time
import os
import inspect
import glob
import yaml
import importlib
import numpy as np
import gym
from gym.envs.registration import load
try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None
try:
    import mpi4py
except ImportError:
    mpi4py = None

from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines.common.policies import FeedForwardPolicy as BasePolicy
from stable_baselines.common.policies import register_policy
from stable_baselines.sac.policies import FeedForwardPolicy as SACPolicy
from stable_baselines.bench import Monitor
from stable_baselines import logger
from stable_baselines import PPO2, A2C, ACER, ACKTR, DQN, HER, SAC
from stable_baselines import OPOLO, SAIL
from stable_baselines import TD3, TD3BCO, TD3DAC, TD3DACFO
from stable_baselines.ddpg import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.common.vec_env import VecFrameStack, SubprocVecEnv, VecNormalize, DummyVecEnv, VecEnv
from stable_baselines.td3.lfd_envs import AbsorbingWrapper
# DDPG and TRPO require MPI to be installed
if mpi4py is None:
    DDPG, TRPO = None, None
    DDPGfD = None
else:
    from stable_baselines import DDPG, DDPGfD, TRPO, TRPOGAIL, TRPOGAIFO

from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, \
    VecFrameStack, SubprocVecEnv
from stable_baselines.common.cmd_util import make_atari_env, make_atari_env_with_log_monitor
from stable_baselines.common import set_global_seeds

ALGOS = {
    'a2c': A2C,
    'acer': ACER,
    'acktr': ACKTR,
    'dqn': DQN,
    'ddpg': DDPG,
    'ddpgfd': DDPGfD,
    'her': HER,
    'sac': SAC,
    'ppo2': PPO2,
    'trpo': TRPO,
    'trpogail': TRPOGAIL,
    'trpogaifo': TRPOGAIFO,
    'td3dac': TD3DAC,
    'td3dacfo': TD3DACFO,
    'td3': TD3,
    'td3bco': TD3BCO,
    'sail': SAIL,
    'opolo': OPOLO,
}


# ================== Custom Policies =================

class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                              layers=[64],
                                              layer_norm=True,
                                              feature_extraction="mlp")


class CustomMlpPolicy(BasePolicy):
    def __init__(self, *args, **kwargs):
        super(CustomMlpPolicy, self).__init__(*args, **kwargs,
                                              layers=[16],
                                              feature_extraction="mlp")


class CustomSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs,
                                              layers=[256, 256],
                                              feature_extraction="mlp")


register_policy('CustomSACPolicy', CustomSACPolicy)
register_policy('CustomDQNPolicy', CustomDQNPolicy)
register_policy('CustomMlpPolicy', CustomMlpPolicy)


def flatten_dict_observations(env):
    assert isinstance(env.observation_space, gym.spaces.Dict)
    keys = env.observation_space.spaces.keys()
    return gym.wrappers.FlattenDictWrapper(env, dict_keys=list(keys))


def get_wrapper_class(hyperparams):
    """
    Get a Gym environment wrapper class specified as a hyper parameter
    "env_wrapper".
    e.g.
    env_wrapper: gym_minigrid.wrappers.FlatObsWrapper

    :param hyperparams: (dict)
    :return: a subclass of gym.Wrapper (class object) you can use to
             create another Gym env giving an original env.
    """

    def get_module_name(fullname):
        return '.'.join(wrapper_name.split('.')[:-1])

    def get_class_name(fullname):
        return wrapper_name.split('.')[-1]

    if 'env_wrapper' in hyperparams.keys():
        wrapper_name = hyperparams.get('env_wrapper')
        wrapper_module = importlib.import_module(get_module_name(wrapper_name))
        return getattr(wrapper_module, get_class_name(wrapper_name))
    else:
        return None

def make_env_with_log_monitor(env_id, rank=0, seed=0, log_dir=None, wrapper_class=None):
    """
    Helper function to multiprocess training
    and log the progress.

    :param env_id: (str)
    :param rank: (int)
    :param seed: (int)
    :param log_dir: (str)
    :param wrapper: (type) a subclass of gym.Wrapper to wrap the original
                    env with
    """
    if log_dir not in ['', None]:
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = None

    def _init():
        set_global_seeds(seed + rank)
        env = gym.make(env_id)

        # Dict observation space is currently not supported.
        # https://github.com/hill-a/stable-baselines/issues/321
        # We allow a Gym env wrapper (a subclass of gym.Wrapper)
        if wrapper_class:
            print(wrapper_class)
            if type(wrapper_class) == list:
                for w in wrapper_class:
                    if w is not None:
                        env = w(env)
            else:
                if wrapper_class is not None:
                    env = wrapper_class(env)
        logpath = None if rank > 1 else os.path.join(log_dir, 'agent0')
        env.seed(seed + rank)
        env = Monitor(env, logpath, allow_early_resets=True)
        return env

    return _init


def make_env(env_id, rank=0, seed=0, log_dir=None, wrapper_class=None):
    """
    Helper function to multiprocess training
    and log the progress.

    :param env_id: (str)
    :param rank: (int)
    :param seed: (int)
    :param log_dir: (str)
    :param wrapper: (type) a subclass of gym.Wrapper to wrap the original
                    env with
    """
    if log_dir is None and log_dir != '':
        log_dir = "/tmp/gym/{}/".format(int(time.time()))
    os.makedirs(log_dir, exist_ok=True)

    def _init():
        set_global_seeds(seed + rank)
        env = gym.make(env_id)

        # Dict observation space is currently not supported.
        # https://github.com/hill-a/stable-baselines/issues/321
        # We allow a Gym env wrapper (a subclass of gym.Wrapper)
        if wrapper_class:
            env = wrapper_class(env)

        env.seed(seed + rank)
        env = Monitor(env, os.path.join(log_dir, str(rank)), allow_early_resets=True)
        return env

    return _init


def create_test_env(env_id, n_envs=1, is_atari=False,
                    stats_path=None, seed=0,
                    log_dir='', should_render=True, hyperparams=None):
    """
    Create environment for testing a trained agent

    :param env_id: (str)
    :param n_envs: (int) number of processes
    :param is_atari: (bool)
    :param stats_path: (str) path to folder containing saved running averaged
    :param seed: (int) Seed for random number generator
    :param log_dir: (str) Where to log rewards
    :param should_render: (bool) For Pybullet env, display the GUI
    :param env_wrapper: (type) A subclass of gym.Wrapper to wrap the original
                        env with
    :param hyperparams: (dict) Additional hyperparams (ex: n_stack)
    :return: (gym.Env)
    """
    # HACK to save logs
    if log_dir is not None:
        os.environ["OPENAI_LOG_FORMAT"] = 'csv'
        os.environ["OPENAI_LOGDIR"] = os.path.abspath(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        logger.configure()

    if hyperparams is None:
        hyperparams = {}

    # Create the environment and wrap it if necessary
    env_wrapper = get_wrapper_class(hyperparams)
    if 'env_wrapper' in hyperparams.keys():
        del hyperparams['env_wrapper']

    if is_atari:
        print("Using Atari wrapper")
        env = make_atari_env(env_id, num_env=n_envs, seed=seed)
        # Frame-stacking with 4 frames
        env = VecFrameStack(env, n_stack=4)
    elif n_envs > 1:
        # start_method = 'spawn' for thread safe
        env = SubprocVecEnv([make_env(env_id, i, seed, log_dir, wrapper_class=env_wrapper) for i in range(n_envs)])
    # Pybullet envs does not follow gym.render() interface
    elif "Bullet" in env_id:
        spec = gym.envs.registry.env_specs[env_id]
        try:
            class_ = load(spec.entry_point)
        except AttributeError:
            # Backward compatibility with gym
            class_ = load(spec._entry_point)
        # HACK: force SubprocVecEnv for Bullet env that does not
        # have a render argument
        render_name = None
        use_subproc = 'renders' not in inspect.getfullargspec(class_.__init__).args
        if not use_subproc:
            render_name = 'renders'
        # Dev branch of pybullet
        # use_subproc = use_subproc and 'render' not in inspect.getfullargspec(class_.__init__).args
        # if not use_subproc and render_name is None:
        #     render_name = 'render'

        # Create the env, with the original kwargs, and the new ones overriding them if needed
        def _init():
            # TODO: fix for pybullet locomotion envs
            env = class_(**{**spec._kwargs}, **{render_name: should_render})
            env.seed(0)
            if log_dir is not None:
                env = Monitor(env, os.path.join(log_dir, "0"), allow_early_resets=True)
            return env

        if use_subproc:
            env = SubprocVecEnv([make_env(env_id, 0, seed, log_dir, wrapper_class=env_wrapper)])
        else:
            env = DummyVecEnv([_init])
    else:
        env = DummyVecEnv([make_env(env_id, 0, seed, log_dir, wrapper_class=env_wrapper)])

    # Load saved stats for normalizing input and rewards
    # And optionally stack frames
    if stats_path is not None:
        if hyperparams['normalize']:
            print("Loading running average")
            print("with params: {}".format(hyperparams['normalize_kwargs']))
            env = VecNormalize(env, training=False, **hyperparams['normalize_kwargs'])
            env.load_running_average(stats_path)

        n_stack = hyperparams.get('frame_stack', 0)
        if n_stack > 0:
            print("Stacking {} frames".format(n_stack))
            env = VecFrameStack(env, n_stack)
    return env


def linear_schedule(initial_value):
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        return progress * initial_value

    return func


def get_trained_models(log_folder):
    """
    :param log_folder: (str) Root log folder
    :return: (dict) Dict representing the trained agent
    """
    algos = os.listdir(log_folder)
    trained_models = {}
    for algo in algos:
        for ext in ['zip', 'pkl']:
            for env_id in glob.glob('{}/{}/*.{}'.format(log_folder, algo, ext)):
                # Retrieve env name
                env_id = env_id.split('/')[-1].split('.{}'.format(ext))[0]
                trained_models['{}-{}'.format(algo, env_id)] = (algo, env_id)
    return trained_models


def get_latest_run_id(log_path, env_id):
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :param log_path: (str) path to log folder
    :param env_id: (str)
    :return: (int) latest run number
    """
    max_run_id = 0
    for path in glob.glob(log_path + "/{}_[0-9]*".format(env_id)):
        file_name = path.split("/")[-1]
        ext = file_name.split("_")[-1]
        if env_id == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id


def get_saved_hyperparams(stats_path, norm_reward=False, test_mode=False):
    """
    :param stats_path: (str)
    :param norm_reward: (bool)
    :param test_mode: (bool)
    :return: (dict, str)
    """
    hyperparams = {}
    if not os.path.isdir(stats_path):
        stats_path = None
    else:
        config_file = os.path.join(stats_path, 'config.yml')
        if os.path.isfile(config_file):
            # Load saved hyperparameters
            with open(os.path.join(stats_path, 'config.yml'), 'r') as f:
                hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)
            hyperparams['normalize'] = hyperparams.get('normalize', False)
        else:
            obs_rms_path = os.path.join(stats_path, 'obs_rms.pkl')
            hyperparams['normalize'] = os.path.isfile(obs_rms_path)

        # Load normalization params
        if hyperparams['normalize']:
            if isinstance(hyperparams['normalize'], str):
                normalize_kwargs = eval(hyperparams['normalize'])
                if test_mode:
                    normalize_kwargs['norm_reward'] = norm_reward
            else:
                normalize_kwargs = {'norm_obs': hyperparams['normalize'], 'norm_reward': norm_reward}
            hyperparams['normalize_kwargs'] = normalize_kwargs
    return hyperparams, stats_path


def find_saved_model(algo, log_path, env_id):
    """
    :param algo: (str)
    :param log_path: (str) Path to the directory with the saved model
    :param env_id: (str)
    :return: (str) Path to the saved model
    """
    model_path, found = None, False
    for ext in ['pkl', 'zip']:
        model_path = "{}/{}.{}".format(log_path, env_id, ext)
        found = os.path.isfile(model_path)
        if found:
            break

    if not found:
        raise ValueError("No model found for {} on {}, path: {}".format(algo, env_id, model_path))
    return model_path


def create_env_(args, hyperparams, n_envs, is_atari, env_wrapper=None, normalize=False, sparse_wrapper=None, normalize_kwargs={}):
    #
    if is_atari:
        if args.verbose > 0:
            print("Using Atari wrapper")
        env = make_atari_env_with_log_monitor(args.env, log_dir=args.log_dir, num_env=n_envs, seed=args.seed)
        # Frame-stacking with 4 frames
        env = VecFrameStack(env, n_stack=4)
    elif 'dqn' in args.algo or 'ddpg' in args.algo:
        env = gym.make(args.env)
        env.seed(args.seed)
        env = Monitor(env, args.log_dir, allow_early_resets=True)
        ## add sparse environment wrapper after the monitor wrapper
        #env = CartPoleWrapper(env)
        if env_wrapper is not None:
            print("Using Predefined Env Wrapper")
            env = env_wrapper(env)
        env = DummyVecEnv([lambda: env])
    else: # ppo2, trpo, td3, sac
        if env_wrapper is not None:
            print("Using Predefined Env Wrapper")
        if n_envs == 1:
            env = make_env_with_log_monitor(
                args.env,
                0,
                args.seed,
                log_dir=args.log_dir,
                wrapper_class=[env_wrapper,AbsorbingWrapper] if 'dac' in args.algo else [env_wrapper])
            env = DummyVecEnv([env])
        else:
            # On most env, SubprocVecEnv does not help and is quite memory hungry
            env = DummyVecEnv([
                make_env_with_log_monitor(
                    args.env,
                    i,
                    args.seed,
                    log_dir=args.log_dir,
                    wrapper_class=[env_wrapper,AbsorbingWrapper] if 'dac' in args.algo else [env_wrapper]) for i in range(n_envs)])
        if normalize:
            if args.verbose > 0:
                if len(normalize_kwargs) > 0:
                    print("Normalization activated: {}".format(normalize_kwargs))
                else:
                    print("Normalizing input and reward")
            env = VecNormalize(env, **normalize_kwargs)
    # Optional Frame-stacking
    if hyperparams.get('frame_stack', False):
        n_stack = hyperparams['frame_stack']
        env = VecFrameStack(env, n_stack)
        print("Stacking {} frames".format(n_stack))
        del hyperparams['frame_stack']
    return env, hyperparams


def create_test_env(args, hyperparams):
    env_wrapper = get_wrapper_class(hyperparams)
    n_envs = hyperparams.get('n_envs', 1)
    print("Creating Testing environment for {} ....".format(args.env))
    if env_wrapper is not None:
        print("Using Predefined Env Wrapper")
    logdir = os.path.join(args.log_dir, 'test')
    if n_envs == 1:
        env = make_env_with_log_monitor(
            env_id,
            0,
            args.seed + 1,
            log_dir=logdir,
            wrapper_class=[env_wrapper,AbsorbingWrapper] if 'dac' in args.algo else [env_wrapper])
        env = env()
    else:
        raise ValueError("Multi environment is not supported yet")
    return env

def create_env(args, hyperparams, is_atari, n_timesteps):
    normalize = False
    normalize_kwargs = {}
    if 'normalize' in hyperparams.keys():
        normalize = hyperparams['normalize']
        if isinstance(normalize, str):
            normalize_kwargs = eval(normalize)
            normalize = True
        del hyperparams['normalize']

    if 'policy_kwargs' in hyperparams.keys():
        hyperparams['policy_kwargs'] = eval(hyperparams['policy_kwargs'])

    # Delete keys so the dict can be passed to the model constructor
    n_envs = 1
    if 'n_envs' in hyperparams.keys():
        n_envs = hyperparams['n_envs']
        del hyperparams['n_envs']
    del hyperparams['n_timesteps']
    # obtain a class object from a wrapper name string in hyperparams
    # and delete the entry
    env_wrapper = get_wrapper_class(hyperparams)
    env, hyperparams = create_env_(
        args,
        hyperparams,
        n_envs,
        is_atari,
        env_wrapper=env_wrapper,
        normalize=normalize,
        normalize_kwargs=normalize_kwargs)
    # Stop env processes to free memory
    if args.optimize_hyperparameters and n_envs > 1:
        env.close()
        #exit()

    # Parse noise string for DDPG and SAC
    if ('td3' in args.algo or 'ddpg' in args.algo or "sac" in args.algo or args.algo in ["sail", "opolo", "bcq", "dac"]) and hyperparams.get('noise_type') is not None:
        noise_type = hyperparams['noise_type'].strip()
        noise_std = hyperparams['noise_std']
        n_actions = env.action_space.shape[0]
        if 'adaptive-param' in noise_type:
            assert 'ddpg' in args.algo, 'Parameter is not supported by SAC'
            hyperparams['param_noise'] = AdaptiveParamNoiseSpec(initial_stddev=noise_std,
                                                                desired_action_stddev=noise_std)
        elif 'normal' in noise_type:
            if 'lin' in noise_type:
                hyperparams['action_noise'] = LinearNormalActionNoise(mean=np.zeros(n_actions),
                                                                      sigma=noise_std * np.ones(n_actions),
                                                                      final_sigma=hyperparams.get('noise_std_final', 0.0) * np.ones(n_actions),
                                                                      max_steps=n_timesteps)
            else:
                hyperparams['action_noise'] = NormalActionNoise(mean=np.zeros(n_actions),
                                                                sigma=noise_std * np.ones(n_actions))
                # hyperparams["batch_action_noise"] = BatchNormalActionNoise(mean=np.zeros(n_actions),
                #                                                 sigma=noise_std*np.ones(n_actions),
                #                                                 hyperparams['kernel_dim'])
        elif 'ornstein-uhlenbeck' in noise_type:
            hyperparams['action_noise'] = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                                       sigma=noise_std * np.ones(n_actions))
        else:
            raise RuntimeError('Unknown noise type "{}"'.format(noise_type))
        print("Applying {} noise with std {}".format(noise_type, noise_std))
        del hyperparams['noise_type']
        if 'noise_std_final' in hyperparams:
            del hyperparams['noise_std_final']
        if 'noise_std' in hyperparams:
            del hyperparams['noise_std']
    return env, hyperparams, normalize
