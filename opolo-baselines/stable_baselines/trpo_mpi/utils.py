import gym
import numpy as np

from stable_baselines.common.vec_env import VecEnv

class SegmentGenerator(object):
    def __init__(self, policy, env, horizon, discriminator, explore_discriminator=None, replay_buffer=None, miner=None, entropy_coeff=0.001, sess=None,config={}):
        """
        Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    
        :param policy: (MLPPolicy) the policy
        :param env: (Gym Environment) the environment
        :param horizon: (int) the number of timesteps to run per batch
        :param discriminator: (TransitionClassifier) the reward predicter from obsevation and action
        :param gail: (bool) Whether we are using this generator for standard trpo or with gail
        :param logp: if use log(expert-prob) to shape the reward
        """
        self.policy = policy
        self.env = env
        self.horizon = horizon
        self.discriminator = discriminator
        self.explore_discriminator = explore_discriminator
        self.miner = miner
        self.entropy_coeff = entropy_coeff
        self.config = config
        self.sess = sess
        self.replay_buffer = replay_buffer

    def traj_segment_generator(self, gail=True):
        """
        Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)

        :param policy: (MLPPolicy) the policy
        :param env: (Gym Environment) the environment
        :param horizon: (int) the number of timesteps to run per batch
        :param reward_giver: (TransitionClassifier) the reward predicter from obsevation and action
        :param gail: (bool) Whether we are using this generator for standard trpo or with gail
        :return: (dict) generator that returns a dict with the following keys:

            - observations: (np.ndarray) observations
            - rewards: (numpy float) rewards (if gail is used it is the predicted reward)
            - true_rewards: (numpy float) if gail is used it is the original reward
            - vpred: (numpy float) action logits
            - dones: (numpy bool) dones (is end of episode, used for logging)
            - episode_starts: (numpy bool)
                True if first timestep of an episode, used for GAE
            - actions: (np.ndarray) actions
            - nextvpred: (numpy float) next action logits
            - ep_rets: (float) cumulated current episode reward
            - ep_lens: (int) the length of the current episode
            - ep_true_rets: (float) the real environment reward
        """
        # Initialize state variables
        step = 0
        action = self.env.action_space.sample()  # not used, just so we have the datatype
        observation = self.env.reset()

        cur_ep_ret = 0  # return in current episode
        current_it_len = 0  # len of current iteration
        current_ep_len = 0 # len of current episode
        cur_ep_true_ret = 0
        ep_true_rets = []
        ep_rets = []  # returns of completed episodes in this segment
        ep_lens = []  # Episode lengths

        # Initialize history arrays
        observations = np.array([observation for _ in range(self.horizon)])
        next_observations = np.array([observation for _ in range(self.horizon)])
        true_rewards = np.zeros(self.horizon, 'float32')
        rewards = np.zeros(self.horizon, 'float32')
        vpreds = np.zeros(self.horizon, 'float32')

        gail_rewards = np.zeros(self.horizon, 'float32') 
        mi_rewards = np.zeros(self.horizon, 'float32')
        entropy_rewards = np.zeros(self.horizon, 'float32')

        episode_starts = np.zeros(self.horizon, 'bool')
        dones = np.zeros(self.horizon, 'bool')
        actions = np.array([action for _ in range(self.horizon)])
        states = self.policy.initial_state
        episode_start = True  # marks if we're on first timestep of an episode
        done = False

        while True:
            action, vpred, states, neglogp = self.policy.step(observation.reshape(-1, *observation.shape), states, done)
            clipped_action = action
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            # Slight weirdness here because we need value function at time T
            # before returning segment [0, T-1] so we get the correct
            # terminal value
            if step > 0 and step % self.horizon == 0:
                yield {
                        "observations": observations,
                        "next_observations": next_observations,
                        "rewards": rewards,
                        "gail_rewards": gail_rewards,
                        "mi_rewards": mi_rewards,
                        "entropy_rewards": entropy_rewards,
                        "dones": dones,
                        "episode_starts": episode_starts,
                        "true_rewards": true_rewards,
                        "vpred": vpreds,
                        "actions": actions, 
                        "nextvpred": vpred[0] * (1 - episode_start),
                        "ep_rets": ep_rets,
                        "ep_lens": ep_lens,
                        "ep_true_rets": ep_true_rets,
                        "total_timestep": current_it_len
                }
                _, vpred, _, _ = self.policy.step(observation.reshape(-1, *observation.shape))
                # Be careful!!! if you change the downstream algorithm to aggregate
                # several of these batches, then be sure to do a deepcopy
                ep_rets = []
                ep_true_rets = []
                ep_lens = []
                # Reset current iteration length
                current_it_len = 0
            i = step % self.horizon
            observations[i] = observation
            vpreds[i] = vpred[0]
            actions[i] = action[0]
            episode_starts[i] = episode_start

            ## clipped_action = action
            ## # Clip the actions to avoid out of bound error
            ## if isinstance(self.env.action_space, gym.spaces.Box):
            ##     clipped_action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

            next_observation, true_reward, done, info = self.env.step(clipped_action[0])
            next_observations[i] = next_observation
            if gail:
                if 'gail' in self.config['shaping_mode']:
                    entropy_reward = self.entropy_coeff * neglogp  
                    if self.explore_discriminator is None: 
                        #gail_reward = self.discriminator.get_reward(observation.reshape(1, -1), clipped_action[0].reshape(1,-1), unscale=False, sess=self.sess)[0]
                        #reward = gail_reward + entropy_reward
                        gail_reward = self.discriminator.get_reward(observation.reshape(1, -1), clipped_action[0].reshape(1,-1), next_observation.reshape(1,-1), sess=self.sess)[0]
                        reward = gail_reward 
                    else:
                        onpolicy_reward = self.discriminator.get_reward(observation.reshape(1, -1), clipped_action[0].reshape(1,-1), unscale=False, sess=self.sess)[0]
                        offpolicy_reward = self.explore_discriminator.get_reward(observation.reshape(1, -1), clipped_action[0].reshape(1,-1), unscale=False, sess=self.sess)[0]
                        reward = 0.5 * ( onpolicy_reward + offpolicy_reward) + entropy_reward
                elif 'gaifo' in self.config['shaping_mode']:
                    reward = self.discriminator.get_reward(observation.reshape(1, -1), next_observation.reshape(1, -1), self.sess)[0]
                elif 'gaifso' in self.config['shaping_mode']:
                    reward = self.discriminator.get_reward(observation.reshape(1, -1), self.sess)[0]
                elif 'midd' in self.config['shaping_mode']:
                    gail_reward = self.discriminator.get_reward(observation.reshape(1, -1), next_observation.reshape(1, -1), self.sess)[0]
                    entropy_reward = self.entropy_coeff * neglogp  
                    mi_reward = self.miner.get_reward(observation.reshape(1, -1), clipped_action[0].reshape(1,-1), next_observation.reshape(1, -1), self.sess)[0]
                    reward = gail_reward + entropy_reward + mi_reward
                else:
                    raise ValueError ('reward mode not recognized: {}'.format(self.config['shaping_mode']))
            else:
                reward = true_reward  
            # save samples to replay buffer if applicable
            if self.replay_buffer is not None:
                self.replay_buffer.add(observation, clipped_action[0], true_reward, next_observation, float(done))

            observation = next_observation
            rewards[i] = reward
            if 'midd' in self.config['shaping_mode']:
                gail_rewards[i] = gail_reward
                entropy_rewards[i] = entropy_reward
                mi_rewards[i] = mi_reward
            true_rewards[i] = true_reward
            dones[i] = done
            episode_start = done

            cur_ep_ret += reward
            cur_ep_true_ret += true_reward
            current_it_len += 1
            current_ep_len += 1
            if done:
                # Retrieve unnormalized reward if using Monitor wrapper
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    if not gail:
                        cur_ep_ret = maybe_ep_info['r']
                    cur_ep_true_ret = maybe_ep_info['r']

                ep_rets.append(cur_ep_ret)
                ep_true_rets.append(cur_ep_true_ret)
                ep_lens.append(current_ep_len)
                cur_ep_ret = 0
                cur_ep_true_ret = 0
                current_ep_len = 0
                if not isinstance(self.env, VecEnv):
                    observation = self.env.reset()
            step += 1


def traj_segment_generator(policy, env, horizon, reward_giver=None, gail=False):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)

    :param policy: (MLPPolicy) the policy
    :param env: (Gym Environment) the environment
    :param horizon: (int) the number of timesteps to run per batch
    :param reward_giver: (TransitionClassifier) the reward predicter from obsevation and action
    :param gail: (bool) Whether we are using this generator for standard trpo or with gail
    :return: (dict) generator that returns a dict with the following keys:

        - observations: (np.ndarray) observations
        - rewards: (numpy float) rewards (if gail is used it is the predicted reward)
        - true_rewards: (numpy float) if gail is used it is the original reward
        - vpred: (numpy float) action logits
        - dones: (numpy bool) dones (is end of episode, used for logging)
        - episode_starts: (numpy bool)
            True if first timestep of an episode, used for GAE
        - actions: (np.ndarray) actions
        - nextvpred: (numpy float) next action logits
        - ep_rets: (float) cumulated current episode reward
        - ep_lens: (int) the length of the current episode
        - ep_true_rets: (float) the real environment reward
    """
    # Check when using GAIL
    assert not (gail and reward_giver is None), "You must pass a reward giver when using GAIL"

    # Initialize state variables
    step = 0
    action = env.action_space.sample()  # not used, just so we have the datatype
    observation = env.reset()

    cur_ep_ret = 0  # return in current episode
    current_it_len = 0  # len of current iteration
    current_ep_len = 0 # len of current episode
    cur_ep_true_ret = 0
    ep_true_rets = []
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # Episode lengths

    # Initialize history arrays
    observations = np.array([observation for _ in range(horizon)])
    true_rewards = np.zeros(horizon, 'float32')
    rewards = np.zeros(horizon, 'float32')

    gail_rewards = np.zeros(horizon, 'float32') 
    mi_rewards = np.zeros(horizon, 'float32')
    entropy_rewards = np.zeros(horizon, 'float32')
    
    vpreds = np.zeros(horizon, 'float32')
    episode_starts = np.zeros(horizon, 'bool')
    dones = np.zeros(horizon, 'bool')
    actions = np.array([action for _ in range(horizon)])
    states = policy.initial_state
    episode_start = True  # marks if we're on first timestep of an episode
    done = False

    while True:
        action, vpred, states, negp = policy.step(observation.reshape(-1, *observation.shape), states, done)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if step > 0 and step % horizon == 0:
            yield {
                    "observations": observations,
                    "rewards": rewards,
                    "gail_rewards": gail_rewards,
                    "mi_rewards": mi_rewards,
                    "entropy_rewards": entropy_rewards,
                    "dones": dones,
                    "episode_starts": episode_starts,
                    "true_rewards": true_rewards,
                    "vpred": vpreds,
                    "actions": actions,
                    "nextvpred": vpred[0] * (1 - episode_start),
                    "ep_rets": ep_rets,
                    "ep_lens": ep_lens,
                    "ep_true_rets": ep_true_rets,
                    "total_timestep": current_it_len
            }
            _, vpred, _, _ = policy.step(observation.reshape(-1, *observation.shape))
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_true_rets = []
            ep_lens = []
            # Reset current iteration length
            current_it_len = 0
        i = step % horizon
        observations[i] = observation
        vpreds[i] = vpred[0]
        actions[i] = action[0]
        # they save unclipped actions as onpolicy samples, but interact with environment using clipped actions -- Judy 
        episode_starts[i] = episode_start

        clipped_action = action
        # Clip the actions to avoid out of bound error
        if isinstance(env.action_space, gym.spaces.Box):
            clipped_action = np.clip(action, env.action_space.low, env.action_space.high)

        if gail:
            reward = reward_giver.get_reward(observation, clipped_action[0])
            observation, true_reward, done, info = env.step(clipped_action[0])
        else:
            observation, reward, done, info = env.step(clipped_action[0])
            true_reward = reward
        rewards[i] = reward
        true_rewards[i] = true_reward
        dones[i] = done
        episode_start = done

        cur_ep_ret += reward
        cur_ep_true_ret += true_reward
        current_it_len += 1
        current_ep_len += 1
        if done:
            # Retrieve unnormalized reward if using Monitor wrapper
            maybe_ep_info = info.get('episode')
            if maybe_ep_info is not None:
                if not gail:
                    cur_ep_ret = maybe_ep_info['r']
                cur_ep_true_ret = maybe_ep_info['r']

            ep_rets.append(cur_ep_ret)
            ep_true_rets.append(cur_ep_true_ret)
            ep_lens.append(current_ep_len)
            cur_ep_ret = 0
            cur_ep_true_ret = 0
            current_ep_len = 0
            if not isinstance(env, VecEnv):
                observation = env.reset()
        step += 1


def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)

    :param seg: (dict) the current segment of the trajectory (see traj_segment_generator return for more information)
    :param gamma: (float) Discount factor
    :param lam: (float) GAE factor
    """
    # last element is only used for last vtarg, but we already zeroed it if last new = 1
    episode_starts = np.append(seg["episode_starts"], False)
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    rew_len = len(seg["rewards"])
    seg["adv"] = np.empty(rew_len, 'float32')
    rewards = seg["rewards"]
    lastgaelam = 0
    for step in reversed(range(rew_len)):
        nonterminal = 1 - float(episode_starts[step + 1])
        delta = rewards[step] + gamma * vpred[step + 1] * nonterminal - vpred[step]
        seg["adv"][step] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def flatten_lists(listoflists):
    """
    Flatten a python list of list

    :param listoflists: (list(list))
    :return: (list)
    """
    return [el for list_ in listoflists for el in list_]
