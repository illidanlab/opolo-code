import gym
import numpy as np

from stable_baselines.common.vec_env import VecEnv

class SegmentGenerator(object):
    def __init__(self, policy, env, horizon, discriminator=None, sess=None,config=None):
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
        self.config = config
        self.lagrange_lambda = 0
        self.lagrange_stepsize=config['lagrange_stepsize']
        self.shift = 0
        self.return_shift = 0
        self.sess = sess

    def set_lagrange_lambda(self, gradient):
        self.lagrange_lambda = self.lagrange_lambda + self.lagrange_stepsize * gradient
        return self.lagrange_lambda


    def traj_segment_generator_dual(
        self,
        ):
        """
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
        # Initialize configs
        if self.config == None:
            gail=False,
            shaping_mode=None,
            lambda1=0.1 # only for pofd
            dual=False
            sparse = False
        else:
            gail=self.config['gail']
            shaping_mode=self.config['shaping_mode']
            lambda0=self.config['pofd_lambda0']  
            lambda1=self.config['pofd_lambda1']  
            dual = self.config['dual']#True
            sparse = self.config['sparse']
            
            #print(gail, discriminator)

        # Check when using GAIL
        assert not (gail and self.discriminator is None), "You must pass a reward giver when using GAIL"


        # Initialize state variables
        step = 0
        action = self.env.action_space.sample()  # not used, just so we have the datatype
        observation = self.env.reset()

        current_it_len = 0  # len of current iteration
        current_ep_len = 0 # len of current episode
        cur_ep_true_ret = 0
        cur_ep_aug_ret = 0
        ep_true_rets = []
        ep_aug_rets = []
        ep_lens = []  # Episode lengths

        # Initialize history arrays
        observations = np.array([observation for _ in range(self.horizon)])
        true_rewards = np.zeros(self.horizon, 'float32')
        aug_rewards = np.zeros(self.horizon, 'float32')

        true_vpreds = np.zeros(self.horizon, 'float32')
        aug_vpreds = np.zeros(self.horizon, 'float32')

        episode_starts = np.zeros(self.horizon, 'bool')
        dones = np.zeros(self.horizon, 'bool')
        actions = np.array([action for _ in range(self.horizon)])
        states = self.policy.initial_state
        episode_start = True  # marks if we're on first timestep of an episode
        done = False

        while True:
            if dual:
                action, aug_vpred, true_vpred, states, _ = self.policy.step(observation.reshape(-1, *observation.shape), states, done)
            else:
                action, true_vpred, states, _ = self.policy.step(observation.reshape(-1, *observation.shape), states, done)
                aug_vpred = true_vpred  
            # Slight weirdness here because we need value function at time T
            # before returning segment [0, T-1] so we get the correct
            # terminal value
            if step > 0 and step % self.horizon == 0:
                yield {
                        "observations": observations,
                        "dones": dones,
                        "episode_starts": episode_starts,
                        "total_timestep": current_it_len,
                        "true_rewards": true_rewards,
                        "actions": actions,
                        "aug_rewards": aug_rewards,
                        "aug_vpred": aug_vpreds,
                        "true_vpred": true_vpreds,
                        "aug_nextvpred": aug_vpred[0] * (1 - episode_start),
                        "true_nextvpred": true_vpred[0] * (1 - episode_start),
                        "ep_aug_rets": ep_aug_rets,
                        "ep_true_rets": ep_true_rets,
                        "ep_lens": ep_lens
                }
                if dual: 
                    _, aug_vpred, true_vpred, _, _ = self.policy.step(observation.reshape(-1, *observation.shape))
                else:
                    _, true_vpred, _, _ = self.policy.step(observation.reshape(-1, *observation.shape))
                    aug_vpred = true_vpred
                # Be careful!!! if you change the downstream algorithm to aggregate
                # several of these batches, then be sure to do a deepcopy
                ep_aug_rets = []
                ep_true_rets = []
                ep_lens = []
                # Reset current iteration length
                current_it_len = 0
            i = step % self.horizon
            observations[i] = observation
            aug_vpreds[i] = aug_vpred[0]
            true_vpreds[i] = true_vpred[0]

            actions[i] = action[0]
            episode_starts[i] = episode_start

            clipped_action = action
            bonus = 0 
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

            ##############################################
            # sample from envrionment
            if gail: 
                observation = np.expand_dims(observation, axis=0) 
            observation, true_reward, done, info = self.env.step(clipped_action[0])
            if sparse != False:
                sparse_reward = info['sparse_reward']
                true_reward = sparse_reward 
            true_reward = lambda0 * true_reward 
            ##############################################

            if gail:
                #print(observation.shape, clipped_action[0].shape)
                if 'gail' in shaping_mode:
                    bonus = lambda1 * self.discriminator.get_reward(observation, clipped_action[0]) 
                    aug_reward = bonus
                elif 'switch2' in shaping_mode:
                    z, p = self.discriminator.get_confidence(observation, clipped_action[0])
                    bonus = lambda1 * np.log(p + 1e-8) - np.log(1 - p + 1e-8) # same as AIRL
                    aug_reward = bonus
                elif 'switch3' in shaping_mode:
                    z, p = self.discriminator.get_confidence(observation, clipped_action[0])
                    bonus = np.log(p + 1e-8) # always negative,  same as GAIL 
                    bonus = lambda1 * bonus
                    aug_reward = bonus
                elif 'switch4' in shaping_mode:
                    z, p = self.discriminator.get_confidence(observation, clipped_action[0])
                    # z equals -log(1 - p + 1e-8)
                    bonus = z * (1-p) - np.log(p + 1e-8) * p # always positive, entropy maximized
                    bonus = lambda1 * bonus
                    aug_reward = bonus
                elif 'switch5' in shaping_mode:
                    z, p = self.discriminator.get_confidence(observation, clipped_action[0])
                    bonus = np.log(1 - p + 1e-8) * p + np.log(p + 1e-8) * (1-p) # always negative, entropy-maximized
                    bonus = lambda1 * bonus
                    aug_reward = bonus
                ##elif 'switch6' in shaping_mode: # similar to switch4, but shift by subtracting max(bounus)
                ##    z, p = self.discriminator.get_confidence(observation, clipped_action[0])
                ##    bonus = np.log(1 - p + 1e-8) * p + np.log(p + 1e-8) * (1-p) # 
                ##    bonus += np.log(0.5)  
                ##    bonus = lambda1 * bonus
                ##    aug_reward = bonus
                elif 'switch7' in shaping_mode: # clip p to be <= 0.5, so that we do not decrease the reward when p --> 1
                    _, p = self.discriminator.get_confidence(observation, clipped_action[0])
                    p = np.clip(p, 1e-8, 0.5)
                    bonus = - np.log(1 - p) * (1 - p) - np.log(p ) * p # 
                    bonus = lambda1 * bonus
                    aug_reward = bonus
                elif 'switch8' in shaping_mode: # section-wise reward function that is always bounded
                    _, p = self.discriminator.get_confidence(observation, clipped_action[0])
                    if p <= 0.5:
                        bonus = - np.log(1 - p + 1e-8) * (1 - p) - np.log(p + 1e-8) * p # 
                    else:
                        bonus = np.log(p + 0.5) + np.log(2) # 
                    bonus = lambda1 * bonus
                    aug_reward = bonus
                elif 'switch9' in shaping_mode: # section-wise reward function that is always bounded
                    _, p = self.discriminator.get_confidence(observation, clipped_action[0])
                    if p <= 0.5:
                        bonus = - np.log(1 - p + 1e-8) * (1 - p) - np.log(p + 1e-8) * p # 
                    else:
                        bonus = - np.log(1.5 - p ) * (1.5 - p) - np.log(p - 0.5 + 1e-8) * ( p - 0.5) + np.log(2)# 
                    bonus = lambda1 * bonus
                    aug_reward = bonus
                else: # GAIL, only use discriminator logit as reward
                    print(shaping_mode)
                    raise ValueError('Shaping mode Not implemented')
            else: # no gail, no reward-shaping
                aug_reward = 0

            ######################################
            # Caution: true_reward is different from intrinsic reward for sparse-case.
            ######################################

            aug_rewards[i] = aug_reward
            true_rewards[i] = true_reward
            dones[i] = done
            episode_start = done
            cur_ep_aug_ret += aug_reward
            cur_ep_true_ret += true_reward
            current_it_len += 1
            current_ep_len += 1
            if done:
                # Retrieve unnormalized reward if using Monitor wrapper
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None and not sparse: 
                    cur_ep_true_ret = maybe_ep_info['r']
                #if sparse == 'type1':
                #    cur_ep_true_ret = float(maybe_ep_info['r'] > self.config['max_score'])
                ep_aug_rets.append(cur_ep_aug_ret)
                ep_true_rets.append(cur_ep_true_ret)
                ep_lens.append(current_ep_len)
                cur_ep_true_ret = 0
                cur_ep_aug_ret = 0
                current_ep_len = 0
                if not isinstance(self.env, VecEnv):
                    observation = self.env.reset()
            step += 1


    def traj_segment_generator_pro(
        self,
        ):
        """
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
        # Initialize configs
        if self.config == None:
            gail=False,
            shaping_mode=None,
            sparse=False,
            lambda1=0.1 # only for pofd
            lambda0=1
            dual=False
        else:
            gail=self.config['gail']
            shaping_mode=self.config['shaping_mode']
            sparse=self.config['sparse']
            lambda1=self.config['pofd_lambda1'] # only for pofd
            lambda0 = self.config['pofd_lambda0']
            dual = self.config.get('dual', False)
            
            #print(gail, discriminator)

        # Check when using GAIL
        assert not (gail and self.discriminator is None), "You must pass a reward giver when using GAIL"


        # Initialize state variables
        step = 0
        action = self.env.action_space.sample()  # not used, just so we have the datatype
        observation = self.env.reset()

        cur_ep_sparse_ret = 0  # return in current episode
        cur_ep_ret = 0  # return in current episode
        current_it_len = 0  # len of current iteration
        current_ep_len = 0 # len of current episode
        cur_ep_true_ret = 0
        cur_ep_logit_ret = 0
        ep_true_rets = []
        ep_logit_rets = []
        ep_sparse_rets = []
        ep_rets = []  # returns of completed episodes in this segment
        ep_lens = []  # Episode lengths

        # Initialize history arrays
        observations = np.array([observation for _ in range(self.horizon)])
        true_rewards = np.zeros(self.horizon, 'float32')
        rewards = np.zeros(self.horizon, 'float32')
        logits = np.zeros(self.horizon, 'float32')
        sparse_rewards = np.zeros(self.horizon, 'float32')

        vpreds = np.zeros(self.horizon, 'float32')
        true_vpreds = np.zeros(self.horizon, 'float32')

        episode_starts = np.zeros(self.horizon, 'bool')
        dones = np.zeros(self.horizon, 'bool')
        actions = np.array([action for _ in range(self.horizon)])
        states = self.policy.initial_state
        episode_start = True  # marks if we're on first timestep of an episode
        done = False

        while True:
            if dual:
                action, vpred, true_vpred, states, _ = self.policy.step(observation.reshape(-1, *observation.shape), states, done)
                #action, aug_vpred, true_vpred, states, _ = self.policy.step(observation.reshape(-1, *observation.shape), states, done)
            else:
                action, vpred, states, _ = self.policy.step(observation.reshape(-1, *observation.shape), states, done)
                true_vpred = vpred
            # Slight weirdness here because we need value function at time T
            # before returning segment [0, T-1] so we get the correct
            # terminal value
            if step > 0 and step % self.horizon == 0:
                yield {
                        "observations": observations,
                        "rewards": rewards,
                        "logits": logits,
                        "dones": dones,
                        "episode_starts": episode_starts,
                        "true_rewards": true_rewards,
                        "sparse_rewards": sparse_rewards,
                        "vpred": vpreds,
                        "true_vpred": true_vpreds,
                        "actions": actions,
                        "nextvpred": vpred[0] * (1 - episode_start),
                        "true_nextvpred": true_vpred[0] * (1 - episode_start),
                        "ep_rets": ep_rets,
                        "ep_lens": ep_lens,
                        "ep_true_rets": ep_true_rets,
                        "ep_logit_rets": ep_logit_rets,
                        "ep_sparse_rets": ep_sparse_rets,
                        "total_timestep": current_it_len
                }
                if dual: 
                    _, vpred, true_vpred, _, _ = self.policy.step(observation.reshape(-1, *observation.shape))
                else:
                    _, vpred, _, _ = self.policy.step(observation.reshape(-1, *observation.shape))
                    true_vpred = vpred
                # Be careful!!! if you change the downstream algorithm to aggregate
                # several of these batches, then be sure to do a deepcopy
                ep_rets = []
                ep_true_rets = []
                ep_logit_rets = []
                ep_lens = []
                # Reset current iteration length
                current_it_len = 0
            i = step % self.horizon
            observations[i] = observation
            vpreds[i] = vpred[0]
            true_vpreds[i] = true_vpred[0]

            actions[i] = action[0]
            episode_starts[i] = episode_start

            clipped_action = action
            sparse_reward = 0
            bonus = 0 
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

            ##############################################
            # sample from envrionment
            if gail: 
                observation = np.expand_dims(observation, axis=0) 
            observation, true_reward, done, info = self.env.step(clipped_action[0])
            if sparse != False:
                sparse_reward = info['sparse_reward']
                intrinsic_reward = sparse_reward
            else:
                intrinsic_reward = true_reward
            ##############################################

            if gail:
                #print(observation.shape, clipped_action[0].shape)
                _, p = self.discriminator.get_confidence(observation, clipped_action[0], sess=self.sess)
                reward = bonus = - np.log(1 - p + 1e-8)
            else: # no gail/pofd, no reward-shaping
                reward = intrinsic_reward
                bonus = 0 


            ######################################
            # Caution: true_reward is different from intrinsic reward for sparse-case.
            ######################################

            rewards[i] = reward
            logits[i] = bonus
            sparse_rewards[i] = sparse_reward
            true_rewards[i] = true_reward
            dones[i] = done
            episode_start = done
            cur_ep_ret += reward
            cur_ep_true_ret += true_reward
            cur_ep_logit_ret += bonus
            cur_ep_sparse_ret += sparse_reward 
            current_it_len += 1
            current_ep_len += 1
            if done:
                # Retrieve unnormalized reward if using Monitor wrapper
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    if not gail and not sparse:
                        # only be triggered for dense-environment, no shaping/gail/pofd 
                        cur_ep_ret = maybe_ep_info['r']
                    cur_ep_true_ret = maybe_ep_info['r']


                ep_rets.append(cur_ep_ret)
                ep_true_rets.append(cur_ep_true_ret)
                ep_logit_rets.append(cur_ep_logit_ret / current_ep_len)
                ep_sparse_rets.append(cur_ep_sparse_ret)
                ep_lens.append(current_ep_len)
                cur_ep_ret = 0
                cur_ep_true_ret = 0
                cur_ep_logit_ret = 0
                cur_ep_sparse_ret = 0
                current_ep_len = 0
                if not isinstance(self.env, VecEnv):
                    observation = self.env.reset()
            step += 1


def traj_segment_generator(
    policy,
    env,
    horizon,
    discriminator=None,
    gail=False,
    pofd=False, 
    reward_shaping=False,
    expert_model=None,
    sparse=False,
    logp=False,
    lambda1=0.1 # only for pofd
    ):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)

    :param policy: (MLPPolicy) the policy
    :param env: (Gym Environment) the environment
    :param horizon: (int) the number of timesteps to run per batch
    :param discriminator: (TransitionClassifier) the reward predicter from obsevation and action
    :param gail: (bool) Whether we are using this generator for standard trpo or with gail
    :param logp: if use log(expert-prob) to shape the reward
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
    assert not (gail and discriminator is None), "You must pass a reward giver when using GAIL"

    # Initialize state variables
    step = 0
    action = env.action_space.sample()  # not used, just so we have the datatype
    observation = env.reset()

    cur_ep_sparse_ret = 0  # return in current episode
    cur_ep_ret = 0  # return in current episode
    current_it_len = 0  # len of current iteration
    current_ep_len = 0 # len of current episode
    cur_ep_true_ret = 0
    cur_ep_logit_ret = 0
    ep_true_rets = []
    ep_logit_rets = []
    ep_sparse_rets = []
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # Episode lengths

    # Initialize history arrays
    observations = np.array([observation for _ in range(horizon)])
    true_rewards = np.zeros(horizon, 'float32')
    rewards = np.zeros(horizon, 'float32')
    sparse_rewards = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    episode_starts = np.zeros(horizon, 'bool')
    dones = np.zeros(horizon, 'bool')
    actions = np.array([action for _ in range(horizon)])
    states = policy.initial_state
    episode_start = True  # marks if we're on first timestep of an episode
    done = False

    while True:
        action, vpred, states, _ = policy.step(observation.reshape(-1, *observation.shape), states, done)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if step > 0 and step % horizon == 0:
            yield {
                    "observations": observations,
                    "rewards": rewards,
                    "dones": dones,
                    "episode_starts": episode_starts,
                    "true_rewards": true_rewards,
                    "sparse_rewards": sparse_rewards,
                    "vpred": vpreds,
                    "actions": actions,
                    "nextvpred": vpred[0] * (1 - episode_start),
                    "ep_rets": ep_rets,
                    "ep_lens": ep_lens,
                    "ep_true_rets": ep_true_rets,
                    "ep_logit_rets": ep_logit_rets,
                    "ep_sparse_rets": ep_sparse_rets,
                    "total_timestep": current_it_len
            }
            _, vpred, _, _ = policy.step(observation.reshape(-1, *observation.shape))
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_true_rets = []
            ep_logit_rets = []
            ep_lens = []
            # Reset current iteration length
            current_it_len = 0
        i = step % horizon
        observations[i] = observation
        vpreds[i] = vpred[0]
        actions[i] = action[0]
        episode_starts[i] = episode_start

        clipped_action = action
        sparse_reward = 0
        bonus = 0 
        # Clip the actions to avoid out of bound error
        if isinstance(env.action_space, gym.spaces.Box):
            clipped_action = np.clip(action, env.action_space.low, env.action_space.high)

        ##############################################
        # sample from envrionment
        if gail: 
            observation = np.expand_dims(observation, axis=0) 
        observation, true_reward, done, info = env.step(clipped_action[0])
        ##############################################

        if sparse != False:
            sparse_reward = info['sparse_reward']

        if gail:
            #print(observation.shape, clipped_action[0].shape)
            bonus = lambda1 * discriminator.get_reward(observation, clipped_action[0]) 

            if pofd: # POfD, r' = env_r + bonus
                if sparse != False:
                    reward = bonus + sparse_reward  
                else:
                    reward = bonus + true_reward  
            else: # GAIL, only use discriminator logit as reward
                reward = bonus   
        elif reward_shaping:
            ######### reward shaping ##########
            # shape reward using the expert policy
            action_prob_for_shaping = expert_model.action_probability(observation)
            #logit = np.clip(action_prob_for_shaping - np.mean(action_prob_for_shaping), 0, 1)
            logit = action_prob_for_shaping 
            bonus = logit[clipped_action[0]]
            if logp:
                bonus = - lambda1 * np.log(1 - bonus + 1e-8) 
            if sparse  != False:
                reward = bonus + sparse_reward 
            else:
                reward = bonus + true_reward
        else: # no gail/pofd, no reward-shaping
            if sparse != False:
                reward = sparse_reward
            else:
                reward = true_reward

        rewards[i] = reward
        sparse_rewards[i] = sparse_reward
        true_rewards[i] = true_reward
        dones[i] = done
        episode_start = done
        cur_ep_ret += reward
        cur_ep_true_ret += true_reward
        cur_ep_logit_ret += bonus
        cur_ep_sparse_ret += sparse_reward 
        current_it_len += 1
        current_ep_len += 1
        if done:
            # Retrieve unnormalized reward if using Monitor wrapper
            maybe_ep_info = info.get('episode')
            if maybe_ep_info is not None:
                if not gail and not sparse and not reward_shaping:
                    # only be triggered for dense-environment, no shaping/gail/pofd 
                    cur_ep_ret = maybe_ep_info['r']

                cur_ep_true_ret = maybe_ep_info['r']

            ep_rets.append(cur_ep_ret)
            ep_true_rets.append(cur_ep_true_ret)
            ep_logit_rets.append(cur_ep_logit_ret / current_ep_len)
            ep_sparse_rets.append(cur_ep_sparse_ret)
            ep_lens.append(current_ep_len)
            cur_ep_ret = 0
            cur_ep_true_ret = 0
            cur_ep_logit_ret = 0
            cur_ep_sparse_ret = 0
            current_ep_len = 0
            if not isinstance(env, VecEnv):
                observation = env.reset()
        step += 1


def calculate_expert_true_vtarg_and_adv(policy, obs, actions, true_rewards, episode_starts, episode_returns, gamma, lam):
    traj_n = int(sum(episode_starts))
    episode_starts = np.append(episode_starts, True)
    rew_len = len(true_rewards)
    true_vpred = policy.true_value(obs) 
    episode_advs = np.empty(traj_n, 'float32')
    i, episode_adv = 0, 0
    for step in reversed(range(rew_len)):
        next_step = (step + 1) % rew_len
        nonterminal = 1 - float(episode_starts[next_step])
        true_delta = true_rewards[step] + gamma * true_vpred[next_step] * nonterminal - true_vpred[step]  
        episode_adv = true_delta + gamma * episode_adv 
        if nonterminal == 0:
            episode_advs[i] = episode_adv 
            episode_adv = 0 ## a new trajectory begins
            i += 1
    
    return episode_advs

#def calculate_expert_true_vtarg_and_adv(policy, obs, actions, true_rewards, episode_starts, gamma, lam):
#    episode_starts = np.append(episode_starts, True)
#    rew_len = len(true_rewards)
#    traj_n = int(np.sum(episode_starts))
#    true_vpred = policy.true_value(obs) 
#    true_adv = np.empty(rew_len, 'float32')
#    episode_adv = np.empty(traj_n, 'float32')
#    true_lastgaelam, i = 0, 0
#    for step in reversed(range(rew_len)):
#        next_step = (step + 1) % rew_len
#        nonterminal = 1 - float(episode_starts[next_step])
#        true_delta = true_rewards[step] + gamma * true_vpred[next_step] * nonterminal - true_vpred[step]  
#        true_adv[step] = true_lastgaelam = true_delta + gamma * lam * nonterminal * true_lastgaelam
#        #if nonterminal == 0 and i < traj_n:
#        if nonterminal == 0:
#            true_lastgaelam = 0 # a new trajectory begins
#            episode_adv[i] = true_adv[step]
#            i += 1
#    
#    #true_adv = true_adv / (np.sum(episode_starts)) 
#    #true_adv = np.sum(true_adv) / rew_len
#
#    #seg["true_tdlamret"] = seg["true_adv"] + seg["true_vpred"]
#    #seg["tdlamret"] = seg["adv"] + seg["vpred"]
#    return episode_adv 


def add_separate_vtarg_and_adv(seg, gamma, lam):
    """
    similar to add_vtarg_and_adv, but get target & advantage for two MDPs, one for r', one for r + r'.
    """
    # last element is only used for last vtarg, but we already zeroed it if last new = 1
    episode_starts = np.append(seg["episode_starts"], False)
    rew_len = len(seg["true_rewards"])

    true_vpred = np.append(seg["true_vpred"], seg["true_nextvpred"])
    vpred = np.append(seg["vpred"], seg["nextvpred"])

    seg["true_adv"] = np.empty(rew_len, 'float32')
    seg["adv"] = np.empty(rew_len, 'float32')

    true_rewards = seg["true_rewards"]
    rewards = seg["rewards"]

    true_lastgaelam, lastgaelam = 0, 0
    for step in reversed(range(rew_len)):
        nonterminal = 1 - float(episode_starts[step + 1])
        true_delta = true_rewards[step] + gamma * true_vpred[step + 1] * nonterminal - true_vpred[step]  
        seg["true_adv"][step] = true_lastgaelam = true_delta + gamma * lam * nonterminal * true_lastgaelam

        delta = rewards[step] + gamma * vpred[step + 1] * nonterminal - vpred[step] # r(s,a) + gamma * V(s')  - V(s) == advantage 
        seg["adv"][step] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam

    seg["true_tdlamret"] = seg["true_adv"] + seg["true_vpred"]
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def add_dual_vtarg_and_adv(seg, gamma, lam):
    """
    similar to add_vtarg_and_adv, but get target & advantage for two MDPs: one for r', one for r
    """
    # last element is only used for last vtarg, but we already zeroed it if last new = 1
    episode_starts = np.append(seg["episode_starts"], False)
    rew_len = len(seg["true_rewards"])

    true_vpred = np.append(seg["true_vpred"], seg["true_nextvpred"])
    aug_vpred = np.append(seg["aug_vpred"], seg["aug_nextvpred"])

    seg["true_adv"] = np.empty(rew_len, 'float32')
    seg["aug_adv"] = np.empty(rew_len, 'float32')

    true_rewards = seg["true_rewards"]
    aug_rewards = seg["aug_rewards"]

    true_lastgaelam, aug_lastgaelam = 0, 0
    for step in reversed(range(rew_len)):
        nonterminal = 1 - float(episode_starts[step + 1])
        true_delta = true_rewards[step] + gamma * true_vpred[step + 1] * nonterminal - true_vpred[step]  
        seg["true_adv"][step] = true_lastgaelam = true_delta + gamma * lam * nonterminal * true_lastgaelam

        aug_delta = aug_rewards[step] + gamma * aug_vpred[step + 1] * nonterminal - aug_vpred[step] # r(s,a) + gamma * V(s')  - V(s) == advantage 
        seg["aug_adv"][step] = aug_lastgaelam = aug_delta + gamma * lam * nonterminal * aug_lastgaelam

    seg["true_tdlamret"] = seg["true_adv"] + seg["true_vpred"]
    seg["aug_tdlamret"] = seg["aug_adv"] + seg["aug_vpred"]


def add_vtarg_and_adv(seg, gamma, lam, shaping_mode=None):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)

    :param seg: (dict) the current segment of the trajectory (see traj_segment_generator return for more information)
    :param gamma: (float) Discount factor
    :param lam: (float) GAE factor
    """
    if shaping_mode is not None and 'return' in shaping_mode: # perform return shaping
        # last element is only used for last vtarg, but we already zeroed it if last new = 1
        episode_starts = np.append(seg["episode_starts"], False)
        vpred = np.append(seg["vpred"], seg["nextvpred"])
        rew_len = len(seg["rewards"])
        seg["adv"] = np.empty(rew_len, 'float32')
        rewards = seg["true_rewards"]
        lastgaelam = 0
        for step in reversed(range(rew_len)):
            nonterminal = 1 - float(episode_starts[step + 1])
            delta = rewards[step] + gamma * vpred[step + 1] * nonterminal - vpred[step] # adv 
            seg["adv"][step] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        seg["tdlamret"] = seg["adv"] + seg["vpred"] # V(s) is the V function for the original MDP
        if 'only' in shaping_mode:
            seg["adv"] = seg["logits"]     # Adv_synthetic(s,a) = Adv(s,a)_original + r'(s,a)
        else:
            seg["adv"] = seg["adv"] + seg["logits"]     # Adv_synthetic(s,a) = Adv(s,a)_original + r'(s,a)
    else: 
        # last element is only used for last vtarg, but we already zeroed it if last new = 1
        episode_starts = np.append(seg["episode_starts"], False)
        vpred = np.append(seg["vpred"], seg["nextvpred"])
        rew_len = len(seg["rewards"])
        seg["adv"] = np.empty(rew_len, 'float32')
        rewards = seg["rewards"]
        lastgaelam = 0
        for step in reversed(range(rew_len)):
            nonterminal = 1 - float(episode_starts[step + 1])
            delta = rewards[step] + gamma * vpred[step + 1] * nonterminal - vpred[step] # adv 
            seg["adv"][step] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        seg["tdlamret"] = seg["adv"] + seg["vpred"]
    


def flatten_lists(listoflists):
    """
    Flatten a python list of list

    :param listoflists: (list(list))
    :return: (list)
    """
    return [el for list_ in listoflists for el in list_]
