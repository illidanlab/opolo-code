import sys
import time
import warnings

import numpy as np
import tensorflow as tf

from stable_baselines.a2c.utils import total_episode_reward_logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.math_util import unscale_action, scale_action
from stable_baselines.deepq.replay_buffer import ReplayBuffer, TrajectoryBuffer, TeacherBuffer
from stable_baselines.ppo2.ppo2 import safe_mean, get_schedule_fn
from stable_baselines.sac.sac import get_vars
from stable_baselines.td3.policies import TD3Policy
from stable_baselines import logger
from stable_baselines.gail.idm import InverseModel

from stable_baselines.gail.dataset.dataset import ExpertDataset


class TD3BCO(OffPolicyRLModel):
    """
    Behavior Cloning from Observation, based on Twin Delayed DDPG (TD3)  -- Judy
    I borrowed the TD3 framework for simplicity.
    Difference from TD3: no critic, only a policy network (for behavior cloning)
    Adds on TD3: inverse dynamic model

    Original paper: https://arxiv.org/pdf/1805.01954.pdf 

    :param policy: (TD3Policy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) the discount factor
    :param learning_rate: (float or callable) learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values and Actor networks)
        it can be a function of the current progress (from 1 to 0)
    :param buffer_size: (int) size of the replay buffer
    :param batch_size: (int) Minibatch size for each gradient update
    :param tau: (float) the soft update coefficient ("polyak update" of the target networks, between 0 and 1)
    :param policy_delay: (int) Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param action_noise: (ActionNoise) the action noise type. Cf DDPG for the different action noise type.
    :param target_policy_noise: (float) Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: (float) Limit for absolute value of target policy smoothing noise.
    :param train_freq: (int) Update the model every `train_freq` steps.
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param gradient_steps: (int) How many gradient update after each step
    :param random_exploration: (float) Probability of taking a random action (as in an epsilon-greedy strategy)
        This is not needed for TD3 normally but can help exploring when using HER + TD3.
        This hack was present in the original OpenAI Baselines repo (DDPG + HER)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        Note: this has no effect on TD3 logging for now
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """
    def __init__(self, policy, env, gamma=0.99, learning_rate=3e-4, buffer_size=50000, 
                 learning_starts=100, train_freq=100, gradient_steps=100, batch_size=128,
                 tau=0.005, policy_delay=2, action_noise=None,
                 target_policy_noise=0.2, target_noise_clip=0.5,
                 random_exploration=0.0, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None,
                 config={}):

        super(TD3BCO, self).__init__(policy=policy, env=env, replay_buffer=None, verbose=verbose,
                                  policy_base=TD3Policy, requires_vec_env=False, policy_kwargs=policy_kwargs,
                                  seed=seed, n_cpu_tf_sess=config.get('n_jobs', n_cpu_tf_sess))

        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.tau = tau
        self.gradient_steps = gradient_steps
        self.gamma = gamma
        self.action_noise = action_noise
        self.random_exploration = random_exploration
        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        self.graph = None
        self.replay_buffer = None
        self.sess = None
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.params = None
        self.summary = None
        self.policy_tf = None
        self.full_tensorboard_log = full_tensorboard_log

        self.obs_target = None
        self.target_policy_tf = None
        self.actions_ph = None
        self.rewards_ph = None
        self.terminals_ph = None
        self.observations_ph = None
        self.action_target = None
        self.next_observations_ph = None
        self.step_ops = None
        self.target_ops = None
        self.infos_names = None
        self.target_params = None
        self.learning_rate_ph = None
        self.processed_obs_ph = None
        self.processed_next_obs_ph = None
        self.policy_out = None
        self.policy_train_op = None
        self.policy_loss = None

        ###############################
        # customized parameters
        ###############################
        self.idm_learning_rate = 3e-4
        self.config = config  
        self.train_freq_alpha = 0.1
        self.idm_gradient_steps = 100 
        self.learning_starts = int(self.train_freq / self.train_freq_alpha)

        if _init_setup_model:
            self.setup_model(config) 

    def _get_pretrain_placeholders(self):
        policy = self.policy_tf
        # Rescale
        policy_out = unscale_action(self.action_space, self.policy_out)
        return policy.obs_ph, self.actions_ph, policy_out


    def initialize_teacher_buffer(self):
        demo_obs, demo_actions, demo_rewards, demo_dones, demo_next_obs, demo_episode_scores = self.expert_dataset.get_transitions()
        episode_lengths = np.where(demo_dones == 1)[0]
        n_samples = len(demo_obs)
        # get episode_score for each demo sample, either 0 or episode-reward
        episode_idx = 0 
        for idx in range(n_samples-1):
            episode_score = demo_episode_scores[episode_idx]
            episode_length = episode_lengths[episode_idx]

            if demo_dones[idx+1] == 1:
                print('{}-th sample, episode_score for demonstration tarjectory: {}'.format(idx, episode_score))
                episode_idx += 1
                assert episode_length - idx >= 0
            scaled_demo_action = scale_action(self.action_space, demo_actions[idx])
            self.teacher_buffer.add(demo_obs[idx], scaled_demo_action, demo_rewards[idx], demo_next_obs[idx], float(demo_dones[idx]))

            if idx % 1000 == 0:
                print("Adding demonstration to the replay buffer, processing {} %  ..".format(float(idx+1) * 100 / n_samples))
        ### add last sample to buffer
        scaled_demo_action = scale_action(self.action_space, demo_actions[-1])
        self.teacher_buffer.add(demo_obs[-1], scaled_demo_action, demo_rewards[-1], demo_next_obs[-1], float(demo_dones[-1]))


    def setup_model(self,config):
        self.expert_data_path = config.get('expert_data_path', None) 
        with SetVerbosity(self.verbose):
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                self.replay_buffer = ReplayBuffer(self.buffer_size)
                self.teacher_buffer = ReplayBuffer(self.buffer_size)

                # build Inverse dynamic model 
                self.inverse_model = InverseModel(
                    self.observation_space,
                    self.action_space,
                    hidden_size=256,
                    reg_coeff=1e-3,
                    learning_rate=self.idm_learning_rate) 

                with tf.variable_scope("input", reuse=False):
                    # Create policy and target TF objects
                    self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                                 **self.policy_kwargs)

                    # Initialize Placeholders
                    self.observations_ph = self.policy_tf.obs_ph
                    # Normalized observation for pixels
                    self.processed_obs_ph = self.policy_tf.processed_obs
                    self.actions_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape,
                                                     name='actions')
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

                    # Initialize Placehoders for Inverse Dynamics Regularization
                    # NOTE: TBD for atari games with CNN policy, we will need to re-scale the idm_obs_ph --Judy
                    self.idm_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_space.shape, name="idm_obs_ph")
                    self.idm_inverse_acs_ph = tf.placeholder(tf.float32, (None,) + self.action_space.shape, name="idm_inverse_acs_ph")


                with tf.variable_scope("model", reuse=False):
                    # Create the policy
                    self.policy_out = policy_out = self.policy_tf.make_actor(self.processed_obs_ph)
                    self.idm_policy_predict = self.policy_tf.make_actor(self.idm_obs_ph, reuse=True)

                with tf.variable_scope("loss", reuse=False):
                    # Policy loss: minimize difference between teacher-inverse-action and predicted-inveres-action
                    self.policy_loss = policy_loss = tf.reduce_mean((self.idm_inverse_acs_ph - self.idm_policy_predict) ** 2)
                   

                    # Policy train op
                    # will be called only every n training steps,
                    # where n is the policy delay
                    policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    policy_train_op = policy_optimizer.minimize(policy_loss, var_list=get_vars('model/pi'))
                    self.policy_train_op = policy_train_op

                    self.step_ops = [self.policy_train_op, self.policy_loss]

                    # Monitor losses and entropy in tensorboard
                    summary_list = []
                    summary_list.append(tf.summary.scalar('policy_loss', self.policy_loss))
                    summary_list.append(tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph)))

                # Retrieve parameters that must be saved
                self.params = get_vars("model")
                # Initialize Variables and target network
                with self.sess.as_default():
                    self.sess.run(tf.global_variables_initializer())

                self.summary = tf.summary.merge(summary_list)

    def _train_policy(self, step, writer, learning_rate):
        # Sample a batch from the teacher buffer
        teacher_batch_obs, _, _, teacher_batch_next_obs, _ = self.teacher_buffer.sample(self.batch_size)
        # get inverse actions based on teacher state transitions
        teacher_inverse_actions = self.inverse_model.get_inverse_action(teacher_batch_obs, teacher_batch_next_obs, sess=self.sess)
        teacher_inverse_actions =  np.array(teacher_inverse_actions).reshape(-1, self.action_space.shape[0])
        feed_dict = {
            self.idm_obs_ph: teacher_batch_obs,
            self.idm_inverse_acs_ph: teacher_inverse_actions,
            self.learning_rate_ph: learning_rate
        }
        step_ops = self.step_ops
        if writer is not None:
            out = self.sess.run([self.summary] + step_ops, feed_dict)
            summary = out.pop(0)
            writer.add_summary(summary, step)
        else:
            self.sess.run(step_ops, feed_dict)
        del teacher_batch_obs, teacher_batch_next_obs

    def should_train(self, step):
        # at first, we only pre-train IDM models
        can_sample = self.teacher_buffer.can_sample(self.batch_size) and self.replay_buffer.can_sample(self.batch_size)
        return can_sample and step > self.learning_starts and step % self.train_freq == 0

    def learn(self, total_timesteps, callback=None,
              log_interval=4, tb_log_name="BCO", reset_num_timesteps=True, replay_wrapper=None):

        self.expert_dataset = ExpertDataset(expert_path=self.expert_data_path, ob_flatten=False)
        print('-'*20 + "expert_data_path: {}".format(self.expert_data_path))
        self.initialize_teacher_buffer()

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        if replay_wrapper is not None:
            self.replay_buffer = replay_wrapper(self.replay_buffer)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:

            self._setup_learn()

            # Transform to callable if needed
            self.learning_rate = get_schedule_fn(self.learning_rate)
            # Initial learning rate
            current_lr = self.learning_rate(1)

            start_time = time.time()
            episode_rewards = [0.0]
            episode_successes = []
            if self.action_noise is not None:
                self.action_noise.reset()
            obs = self.env.reset()
            n_updates = 0
            for step in range(total_timesteps):
                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break

                # Before training starts, randomly sample actions
                # from a uniform distribution for better exploration.
                # Afterwards, use the learned policy
                # if random_exploration is set to 0 (normal setting)
                if step <= self.learning_starts or np.random.rand() < self.random_exploration:
                    # actions sampled from action space are from range specific to the environment
                    # but algorithm operates on tanh-squashed actions therefore simple scaling is used
                    unscaled_action = self.env.action_space.sample()
                    action = scale_action(self.action_space, unscaled_action)
                else:
                    action = self.policy_tf.step(obs[None]).flatten()
                    # Add noise to the action, as the policy
                    # is deterministic, this is required for exploration
                    if self.action_noise is not None:
                        action = np.clip(action + self.action_noise(), -1, 1)
                    # Rescale from [-1, 1] to the correct bounds
                    unscaled_action = unscale_action(self.action_space, action)

                assert action.shape == self.env.action_space.shape

                new_obs, reward, done, info = self.env.step(unscaled_action)

                # Store transition in the replay buffer.
                self.replay_buffer.add(obs, action, reward, new_obs, float(done))
                obs = new_obs

                # Retrieve reward and episode length if using Monitor wrapper
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    self.ep_info_buf.extend([maybe_ep_info])

                if writer is not None:
                    # Write reward per episode to tensorboard
                    ep_reward = np.array([reward]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    total_episode_reward_logger(self.episode_reward, ep_reward,
                                                ep_done, writer, self.num_timesteps)

                if step == self.learning_starts:
                    ####### Pre-demonstration phase #########
                    print('Pre-training IDM model')
                    self.inverse_model.train_idm(
                        writer,
                        logger,
                        step,
                        int(self.idm_gradient_steps / self.train_freq_alpha),
                        self.replay_buffer,
                        self.batch_size,
                        self.sess
                    )
                    print('IDM pre-training complete.')
                ####### Post-demonstration phase #########
                if self.should_train(step):
                    ## Train Inverse Dynamic Model
                    self.inverse_model.train_idm(
                        writer,
                        logger,
                        step,
                        self.idm_gradient_steps,
                        self.replay_buffer,
                        self.batch_size,
                        self.sess
                    )

                    ## Train policy 
                    for grad_step in range(self.gradient_steps):
                        # Break if the warmup phase is not over
                        # or if there are not enough samples in the replay buffer
                        n_updates += 1
                        # Compute current learning_rate
                        frac = 1.0 - step / total_timesteps
                        current_lr = self.learning_rate(frac)
                        self._train_policy(step, writer, current_lr)

                episode_rewards[-1] += reward
                if done:
                    if self.action_noise is not None:
                        self.action_noise.reset()
                    if not isinstance(self.env, VecEnv):
                        obs = self.env.reset()
                    episode_rewards.append(0.0)

                    maybe_is_success = info.get('is_success')
                    if maybe_is_success is not None:
                        episode_successes.append(float(maybe_is_success))

                if len(episode_rewards[-101:-1]) == 0:
                    mean_reward = -np.inf
                else:
                    mean_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                num_episodes = len(episode_rewards)
                self.num_timesteps += 1
                # Display training infos
                if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    fps = int(step / (time.time() - start_time))
                    logger.logkv("episodes", num_episodes)
                    logger.logkv("mean 100 episode reward", mean_reward)
                    if len(self.ep_info_buf) > 0 and len(self.ep_info_buf[0]) > 0:
                        logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in self.ep_info_buf]))
                        logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in self.ep_info_buf]))
                    logger.logkv("n_updates", n_updates)
                    logger.logkv("current_lr", current_lr)
                    logger.logkv("fps", fps)
                    logger.logkv('time_elapsed', int(time.time() - start_time))
                    if len(episode_successes) > 0:
                        logger.logkv("success rate", np.mean(episode_successes[-100:]))
                    logger.logkv("total timesteps", self.num_timesteps)
                    logger.dumpkvs()
            return self

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        _ = np.array(observation)
        if actions is not None:
            raise ValueError("Error: TD3 does not have action probabilities.")

        # here there are no action probabilities, as DDPG does not use a probability distribution
        warnings.warn("Warning: action probability is meaningless for TD3. Returning None")
        return None

    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions = self.policy_tf.step(observation)

        if self.action_noise is not None and not deterministic:
            actions = np.clip(actions + self.action_noise(), -1, 1)

        actions = actions.reshape((-1,) + self.action_space.shape)  # reshape to the correct action shape
        actions = unscale_action(self.action_space, actions)  # scale the output for the prediction

        if not vectorized_env:
            actions = actions[0]

        return actions, None

    def get_parameter_list(self):
        return (self.params)

    def save(self, save_path, cloudpickle=False):
        data = {
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "batch_size": self.batch_size,
            "tau": self.tau,
            # Should we also store the replay buffer?
            # this may lead to high memory usage
            # with all transition inside
            # "replay_buffer": self.replay_buffer
            "policy_delay": self.policy_delay,
            "target_noise_clip": self.target_noise_clip,
            "target_policy_noise": self.target_policy_noise,
            "gamma": self.gamma,
            "verbose": self.verbose,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "action_noise": self.action_noise,
            "random_exploration": self.random_exploration,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)
