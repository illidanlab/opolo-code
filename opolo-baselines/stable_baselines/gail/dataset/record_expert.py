import os
import warnings

import cv2
import numpy as np
from gym import spaces

from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.vec_env import VecEnv, VecFrameStack
from stable_baselines.common.base_class import _UnvecWrapper
from stable_baselines.results_plotter import load_results, ts2xy


def generate_expert_traj_mujoco(model, save_path=None, env=None, n_timesteps=0, save_image=True,
                         n_episodes=100, optimal_score=-1):
    """
    Train expert controller (if needed) and record expert trajectories.
    :return: (dict) the generated expert trajectories.
    """

    # Retrieve the environment using the RL model
    if env is None and isinstance(model, BaseRLModel):
        env = model.get_env()

    assert env is not None, "You must set the env in the model or pass it to the function."

    is_vec_env = False
    if isinstance(env, VecEnv) and not isinstance(env, _UnvecWrapper):
        is_vec_env = True
        if env.num_envs > 1:
            warnings.warn("You are using multiple envs, only the data from the first one will be recorded.")

    # Sanity check
    assert (isinstance(env.observation_space, spaces.Box) or
            isinstance(env.observation_space, spaces.Discrete)), "Observation space type not supported"

    assert (isinstance(env.action_space, spaces.Box) or
            isinstance(env.action_space, spaces.Discrete)), "Action space type not supported"

    # Check if we need to record images
    obs_space = env.observation_space

    if n_timesteps > 0 and isinstance(model, BaseRLModel):
        model.learn(n_timesteps)

    actions = []
    observations = []
    next_observations = []
    rewards = []
    episode_returns = np.zeros((n_episodes,))
    episode_starts = []

    tmp_actions = []
    tmp_observations = []
    tmp_next_observations = []
    tmp_rewards = []
    tmp_episode_starts = []
    reward_sum = 0.0

    ep_idx = 0
    obs = env.reset()
    episode_starts.append(True)
    reward_sum = 0.0
    idx = 0
    # state and mask for recurrent policies
    state, mask = None, None

    if is_vec_env:
        mask = [True for _ in range(env.num_envs)]

    while ep_idx < n_episodes:
        tmp_observations.append(obs)

        if isinstance(model, BaseRLModel):
            action, state = model.predict(obs, state=state, mask=mask)
        else:
            action = model(obs)

        obs, reward, done, infos = env.step(action)
        #print(reward)

        # Use only first env
        if is_vec_env:
            mask = [done[0] for _ in range(env.num_envs)]
            action = np.array([action[0]])
            reward = np.array([reward[0]])
            done = np.array([done[0]])
            infos = np.array([infos[0]])

        tmp_next_observations.append(obs)
        tmp_actions.append(action)
        tmp_rewards.append(reward)
        tmp_episode_starts.append(done)
        reward_sum += reward

        idx += 1

        if done:
            episode_info = infos[0].get('episode')
            if episode_info is not None:
                episode_score = episode_info['r']
                print("Episode score: {:.2f}".format(episode_score))
            else:
                episode_score = None
            if not is_vec_env:
                obs = env.reset()
                # Reset the state in case of a recurrent policy
                state = None

            #qualify = (episode_score is not None) and (episode_score > optimal_score * 0.9) and (episode_score < optimal_score * 1.2)
            #qualify = (episode_score is not None) and (episode_score > optimal_score )#* 0.9) and (episode_score < optimal_score * 1.2)
            qualify = (episode_score is not None) and (episode_score > optimal_score * 0.9) and (episode_score < optimal_score * 3)
            if qualify:
                print("Find expert trajectory with reward {:.2f}".format(episode_score))
                episode_returns[ep_idx] = reward_sum
                actions += tmp_actions 
                rewards += tmp_rewards
                observations += tmp_observations
                next_observations += tmp_next_observations
                episode_starts += tmp_episode_starts
                ep_idx += 1
            else:
                print("Skipping trajectory with reward {}".format(episode_score))

            reward_sum = 0.0
            del tmp_actions, tmp_rewards, tmp_observations, tmp_next_observations, tmp_episode_starts 
            tmp_actions, tmp_rewards, tmp_observations, tmp_next_observations, tmp_episode_starts  = [], [], [], [], []


    if isinstance(env.observation_space, spaces.Box):
        observations = np.concatenate(observations).reshape((-1,) + env.observation_space.shape)
        next_observations = np.concatenate(next_observations).reshape((-1,) + env.observation_space.shape)
    elif isinstance(env.observation_space, spaces.Discrete):
        observations = np.array(observations).reshape((-1, 1))
        next_observations = np.array(next_observations).reshape((-1, 1))

    if isinstance(env.action_space, spaces.Box):
        actions = np.concatenate(actions).reshape((-1,) + env.action_space.shape)
    elif isinstance(env.action_space, spaces.Discrete):
        actions = np.array(actions).reshape((-1, 1))

    rewards = np.array(rewards)
    episode_starts = np.array(episode_starts[:-1])

    assert len(observations) == len(actions) and len(next_observations) == len(actions)

    numpy_dict = {
        'actions': actions,
        'obs': observations,
        'next_obs': next_observations,
        'rewards': rewards,
        'episode_returns': episode_returns,
        'episode_starts': episode_starts
    }

    for key, val in numpy_dict.items():
        print(key, val.shape)

    if save_path is not None:
        np.savez(save_path, **numpy_dict)

    env.close()

    return numpy_dict



def generate_expert_traj(model, save_path=None, env=None, n_timesteps=0, save_image=False,
                         n_episodes=1, optimal_score=-1, image_folder='recorded_images', is_atari=False):
    """
    Train expert controller (if needed) and record expert trajectories.

    .. note::

        only Box and Discrete spaces are supported for now.

    :param model: (RL model or callable) The expert model, if it needs to be trained,
        then you need to pass ``n_timesteps > 0``.
    :param save_path: (str) Path without the extension where the expert dataset will be saved
        (ex: 'expert_cartpole' -> creates 'expert_cartpole.npz').
        If not specified, it will not save, and just return the generated expert trajectories.
        This parameter must be specified for image-based environments.
    :param env: (gym.Env) The environment, if not defined then it tries to use the model
        environment.
    :param n_timesteps: (int) Number of training timesteps
    :param n_episodes: (int) Number of trajectories (episodes) to record
    :param image_folder: (str) When using images, folder that will be used to record images.
    :return: (dict) the generated expert trajectories.
    """

    # Retrieve the environment using the RL model
    if env is None and isinstance(model, BaseRLModel):
        env = model.get_env()

    assert env is not None, "You must set the env in the model or pass it to the function."

    is_vec_env = False
    if isinstance(env, VecEnv) and not isinstance(env, _UnvecWrapper):
        is_vec_env = True
        if env.num_envs > 1:
            warnings.warn("You are using multiple envs, only the data from the first one will be recorded.")

    # Sanity check
    assert (isinstance(env.observation_space, spaces.Box) or
            isinstance(env.observation_space, spaces.Discrete)), "Observation space type not supported"

    assert (isinstance(env.action_space, spaces.Box) or
            isinstance(env.action_space, spaces.Discrete)), "Action space type not supported"

    # Check if we need to record images
    obs_space = env.observation_space
    record_images = len(obs_space.shape) == 3 and obs_space.shape[-1] in [1, 3, 4] \
                    and obs_space.dtype == np.uint8 and save_image
                    
    if record_images and save_path is None:
        warnings.warn("Observations are images but no save path was specified, so will save in numpy archive; "
                      "this can lead to higher memory usage.")
        record_images = False

    if not record_images and len(obs_space.shape) == 3 and obs_space.dtype == np.uint8:
        warnings.warn("The observations looks like images (shape = {}) "
                      "but the number of channel > 4, so it will be saved in the numpy archive "
                      "which can lead to high memory usage".format(obs_space.shape))

    image_ext = 'jpg'
    if record_images:
        # We save images as jpg or png, that have only 3/4 color channels
        if isinstance(env, VecFrameStack) and env.n_stack == 4:
            # assert env.n_stack < 5, "The current data recorder does no support"\
            #                          "VecFrameStack with n_stack > 4"
            image_ext = 'png'

        folder_path = os.path.dirname(save_path)
        image_folder = os.path.join(folder_path, image_folder)
        os.makedirs(image_folder, exist_ok=True)
        print("=" * 10)
        print("Images will be recorded to {}/".format(image_folder))
        print("Image shape: {}".format(obs_space.shape))
        print("=" * 10)

    if n_timesteps > 0 and isinstance(model, BaseRLModel):
        model.learn(n_timesteps)

    actions = []
    observations = []
    last_observations = []
    rewards = []
    episode_returns = np.zeros((n_episodes,))
    episode_starts = []

    tmp_actions = []
    tmp_observations = []
    tmp_rewards = []
    tmp_episode_starts = []
    reward_sum = 0.0

    ep_idx = 0
    obs = env.reset()
    episode_starts.append(True)
    reward_sum = 0.0
    idx = 0
    # state and mask for recurrent policies
    state, mask = None, None

    if is_vec_env:
        mask = [True for _ in range(env.num_envs)]

    while ep_idx < n_episodes:
        if record_images:
            image_path = os.path.join(image_folder, "{}.{}".format(idx, image_ext))
            obs_ = obs[0] if is_vec_env else obs
            # Convert from RGB to BGR
            # which is the format OpenCV expect
            if obs_.shape[-1] == 3:
                obs_ = cv2.cvtColor(obs_, cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_path, obs_)
            #observations.append(image_path)
            tmp_observations.append(image_path)
        else:
            #observations.append(obs)
            tmp_observations.append(obs)

        if isinstance(model, BaseRLModel):
            action, state = model.predict(obs, state=state, mask=mask)
        else:
            action = model(obs)

        obs, reward, done, infos = env.step(action)

        # Use only first env
        if is_vec_env:
            mask = [done[0] for _ in range(env.num_envs)]
            action = np.array([action[0]])
            reward = np.array([reward[0]])
            done = np.array([done[0]])
            infos = np.array([infos[0]])

        #actions.append(action)
        #rewards.append(reward)
        #episode_starts.append(done)
        #reward_sum += reward

        tmp_actions.append(action)
        tmp_rewards.append(reward)
        tmp_episode_starts.append(done)
        reward_sum += reward

        idx += 1
        episode_score = infos[0].get('episode', {})
        episode_score = episode_score.get('r', None)
        if episode_score is not None: # end of episode
            print("Episode score: {:.2f}".format(episode_score))
            if done:
                if not is_vec_env:
                    obs = env.reset()
                    # Reset the state in case of a recurrent policy
                    state = None

                qualify = episode_score is not None and episode_score >= optimal_score
                if qualify:
                    print("Find expert trajectory with reward {:.2f}".format(episode_score))
                    episode_returns[ep_idx] = episode_score
                    actions += tmp_actions 
                    rewards += tmp_rewards
                    observations += tmp_observations
                    episode_starts += tmp_episode_starts
                    ep_idx += 1
                else:
                    print("Skipping trajectory with reward {}".format(episode_score))
                last_observations = [obs]

                reward_sum = 0.0
                del tmp_actions, tmp_rewards, tmp_observations, tmp_episode_starts 
                tmp_actions, tmp_rewards, tmp_observations, tmp_episode_starts  = [], [], [], []


    if isinstance(env.observation_space, spaces.Box) and not record_images:
        observations = np.concatenate(observations).reshape((-1,) + env.observation_space.shape)
    elif isinstance(env.observation_space, spaces.Discrete):
        observations = np.array(observations).reshape((-1, 1))
    elif record_images:
        observations = np.array(observations)

    if isinstance(env.action_space, spaces.Box):
        actions = np.concatenate(actions).reshape((-1,) + env.action_space.shape)
    elif isinstance(env.action_space, spaces.Discrete):
        actions = np.array(actions).reshape((-1, 1))

    rewards = np.array(rewards)
    episode_starts = np.array(episode_starts[:-1])

    numpy_dict = {
        'actions': actions,
        'obs': observations,
        'last_observations': np.array(last_observations).reshape((-1,) + env.observation_space.shape),
        'rewards': rewards,
        'episode_returns': episode_returns,
        'episode_starts': episode_starts
    }

    for key, val in numpy_dict.items():
        print(key, val.shape)

    if save_path is not None:
        np.savez(save_path, **numpy_dict)
        print("Data saved to {}".format(save_path))

    env.close()

    return numpy_dict

