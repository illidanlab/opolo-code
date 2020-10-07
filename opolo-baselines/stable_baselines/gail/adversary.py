"""
Reference: https://github.com/openai/imitation
I follow the architecture from the official repository
"""
import gym
import tensorflow as tf
import numpy as np

from stable_baselines.common.mpi_running_mean_std import RunningMeanStd
from stable_baselines.common.math_util import unscale_action, scale_action
from stable_baselines.common import tf_util as tf_util
from stable_baselines.common import fmt_row  
import random
from mpi4py import MPI


def logsigmoid(input_tensor):
    """
    Equivalent to tf.log(tf.sigmoid(a))

    :param input_tensor: (tf.Tensor)
    :return: (tf.Tensor)
    """
    return -tf.nn.softplus(-input_tensor)


def logit_bernoulli_entropy(logits):
    """
    Reference:
    https://github.com/openai/imitation/blob/99fbccf3e060b6e6c739bdf209758620fcdefd3c/policyopt/thutil.py#L48-L51

    :param logits: (tf.Tensor) the logits
    :return: (tf.Tensor) the Bernoulli entropy
    """
    ent = (1. - tf.nn.sigmoid(logits)) * logits - logsigmoid(logits)
    return ent


def allmean(arr, nworkers):
    assert isinstance(arr, np.ndarray)
    out = np.empty_like(arr)
    MPI.COMM_WORLD.Allreduce(arr, out, op=MPI.SUM)
    out /= nworkers
    return out

class DiscriminatorCalssifier(object):
    def __init__(self, env, observation_space, action_space, hidden_size=256,
                 entcoeff=0.001, gradcoeff=10,scope="adversary", normalize=True):
        """
        Reward regression from observations and transitions

        :param observation_space: (gym.spaces)
        :param action_space: (gym.spaces)
        :param hidden_size: ([int]) the hidden dimension for the MLP
        :param entcoeff: (float) the entropy loss weight
        :param scope: (str) tensorflow variable scope
        :param normalize: (bool) Whether to normalize the reward or not
        """
        # TODO: support images properly (using a CNN)
        self.env = env
        self.scope = scope
        self.allmean = allmean
        self.observation_shape = observation_space.shape
        self.actions_shape = action_space.shape
        self.action_space = action_space
        self.observation_space = observation_space 

        if isinstance(action_space, gym.spaces.Box):
            # Continuous action space
            self.discrete_actions = False
            self.n_actions = action_space.shape[0]
        elif isinstance(action_space, gym.spaces.Discrete):
            self.n_actions = action_space.n
            self.discrete_actions = True
        else:
            raise ValueError('Action space not supported: {}'.format(action_space))

        self.hidden_size = hidden_size
        self.normalize = normalize
        self.obs_rms = None

        # Placeholders
        self.generator_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape,
                                               name="gail_observations_ph")
        self.generator_acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                               name="gail_actions_ph")
        self.expert_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape,
                                            name="gail_expert_observations_ph")
        self.expert_acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                            name="gail_expert_actions_ph")
        self.d_learning_rate_ph = tf.placeholder(tf.float32, [], name="d_learning_rate_ph")
        # Build graph
        generator_input = self.flatten(self.generator_obs_ph, self.generator_acs_ph, reuse=False)
        generator_input = tf.stop_gradient(generator_input)
        generator_logits = self.build_GAN_graph(generator_input, reuse=False)

        expert_input = self.flatten(self.expert_obs_ph, self.expert_acs_ph, reuse=True)
        expert_logits = self.build_GAN_graph(expert_input, reuse=True)

        # Build accuracy
        generator_acc = tf.reduce_mean(tf.cast(tf.nn.sigmoid(generator_logits) < 0.5, tf.float32))
        expert_acc = tf.reduce_mean(tf.cast(tf.nn.sigmoid(expert_logits) > 0.5, tf.float32))
        # Build regression loss
        # let x = logits, z = targets.
        # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        sample_generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits,
                                                                 labels=tf.zeros_like(generator_logits))
        generator_loss = tf.reduce_mean(sample_generator_loss)
        sample_expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits, labels=tf.ones_like(expert_logits))
        expert_loss = tf.reduce_mean(sample_expert_loss)
        # Build entropy loss
        logits = tf.concat([generator_logits, expert_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        entropy_loss = -entcoeff * entropy
        # Build Gradient-penalty loss
        alpha_shape = self.observation_shape[0] + self.n_actions
        alpha = np.random.uniform(size=(1, alpha_shape))
        #inter = tf.multiply(generator_input ) + tf.multiply(expert_input, 1 - alpha) 
        inter = alpha * generator_input + (1 - alpha) * expert_input 
        with tf.GradientTape() as tape2:
            tape2.watch(inter)
            inter_output = self.build_GAN_graph(inter, reuse=True)
            grad = tape2.gradient(inter_output, [inter])[0]
            grad = tf.pow(tf.norm(grad, axis=-1) - 1, 2)
            grad = tf.reduce_mean(grad)
        grad_penalty = gradcoeff * grad

        # Loss + Accuracy terms
        self.losses = [generator_loss, expert_loss, entropy, entropy_loss, generator_acc, expert_acc, grad_penalty]
        self.loss_name = ["generator_loss", "expert_loss", "entropy", "entropy_loss", "generator_acc", "expert_acc", "grad_penalty_loss"]
        self.total_loss = generator_loss + expert_loss + entropy_loss + grad_penalty
        self.sample_loss = [sample_expert_loss, sample_generator_loss]
        # Build Reward for policy
        p = tf.clip_by_value(tf.nn.sigmoid(generator_logits), 0.01, 0.99)
        self.reward_op = -tf.log(1 - p)
        self.original_reward_op = tf.log(p)
        self.confidence_op = tf.nn.sigmoid(generator_logits) 
        # trainable var list
        var_list = self.get_trainable_variables()

        gail_optimizer = tf.train.AdamOptimizer(learning_rate=self.d_learning_rate_ph)
        self.train_op=gail_optimizer.minimize(self.total_loss, var_list=var_list)
        # Log discriminator scalars for debugging purposes
        gail_scalar_summaries = []
        for i, loss_name in enumerate(self.loss_name):
            i = tf.summary.scalar(loss_name, self.losses[i])
            gail_scalar_summaries.append(i)
        self.gail_summary = tf.summary.merge(gail_scalar_summaries)
        self.lossandgrad = tf_util.function(
            [self.generator_obs_ph, self.generator_acs_ph, self.expert_obs_ph, self.expert_acs_ph],
            [self.gail_summary] + self.losses + [tf_util.flatgrad(self.total_loss, var_list)])


    def flatten(self, obs_ph, acs_ph, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            if self.normalize:
                with tf.variable_scope("obfilter"):
                    self.obs_rms = RunningMeanStd(shape=self.observation_shape)
                obs = (obs_ph - self.obs_rms.mean) / self.obs_rms.std
            else:
                obs = obs_ph

            if self.discrete_actions:
                one_hot_actions = tf.one_hot(acs_ph, self.n_actions)
                actions_ph = tf.cast(one_hot_actions, tf.float32)
            else:
                actions_ph = acs_ph

            ob_fl = tf.contrib.layers.flatten(obs)
            act_fl = tf.contrib.layers.flatten(actions_ph)
            flatten_input = tf.concat([ob_fl, act_fl], axis=1)  # concatenate the two input -> form a transition
            return flatten_input

    def build_GAN_graph(self, inputs, reuse=True):
        """
        build the graph, where observations and actions are all flattened,
        in order to address image observations.

        :param obs_ph: (tf.Tensor) the observation placeholder
        :param acs_ph: (tf.Tensor) the action placeholder
        :param reuse: (bool)
        :return: (tf.Tensor) the graph output

        Implement GAIL Discriminator the same as DAC paper.
        Paper:
        https://openreview.net/pdf?id=Hk4fpoA5Km  
        Code:
        https://github.com/google-research/google-research/blob/562c7c6ef959cb3cb382b1b660ccc45e8f5289c4/dac/gail.py 
        hidden size = 256
        """

        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            p_h1 = tf.contrib.layers.fully_connected(inputs, self.hidden_size, activation_fn=tf.nn.tanh)
            p_h2 = tf.contrib.layers.fully_connected(p_h1, self.hidden_size, activation_fn=tf.nn.tanh)
            logits = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.identity) 
        return logits 


    def get_trainable_variables(self):
        """
        Get all the trainable variables from the graph

        :return: ([tf.Tensor]) the variables
        """
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_original_reward(self, obs, actions, unscale=True, sess=None):
        """
        Predict the reward using the observation and action

        :param obs: (tf.Tensor or np.ndarray) the observation
        :param actions: (tf.Tensor or np.ndarray) the action
        :return: (np.ndarray) the reward
        """
        if sess is None:
            sess = tf.get_default_session()
        
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, 0)
        elif len(actions.shape) == 0:
            # one discrete action
            actions = np.expand_dims(actions, 0)

        # unscale action
        actions = unscale_action(self.action_space, actions)
        feed_dict = {self.generator_obs_ph: obs, self.generator_acs_ph: actions}
        reward = sess.run(self.original_reward_op, feed_dict)
        return reward


    def get_reward(self, obs, actions, unscale=True, sess=None):
        """
        Predict the reward using the observation and action

        :param obs: (tf.Tensor or np.ndarray) the observation
        :param actions: (tf.Tensor or np.ndarray) the action
        :return: (np.ndarray) the reward
        """
        if sess is None:
            sess = tf.get_default_session()
        
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, 0)
        elif len(actions.shape) == 0:
            # one discrete action
            actions = np.expand_dims(actions, 0)

        # unscale action
        if unscale:
            actions = unscale_action(self.action_space, actions)
        feed_dict = {self.generator_obs_ph: obs, self.generator_acs_ph: actions}
        reward = sess.run(self.reward_op, feed_dict)
        return reward


    def get_confidence(self, obs, actions, sess=None):
        """
        Predict the reward using the observation and action

        :param obs: (tf.Tensor or np.ndarray) the observation
        :param actions: (tf.Tensor or np.ndarray) the action
        :return: (np.ndarray) the reward
        """
        if sess is None:
            sess = tf.get_default_session()
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, 0)
        elif len(actions.shape) == 0:
            # one discrete action
            actions = np.expand_dims(actions, 0)

        feed_dict = {self.generator_obs_ph: obs, self.generator_acs_ph: actions}
        reward, confidence = sess.run([self.reward_op, self.confidence_op], feed_dict)
        return reward, confidence


    def generate_onpolicy_data(self, teacher_buffer, segs, batch_size, sess):
        observations, actions = segs["observations"], segs["actions"]
        # sample teacher data
        expert_batch = teacher_buffer.sample(batch_size)
        # each sample consists of s, a, r, s', if_done
        ob_expert, ac_expert = expert_batch[0], expert_batch[1]
        # clip teacher actions
        ac_expert = np.clip(ac_expert, self.env.action_space.low, self.env.action_space.high)

        # sample onpoliy data (NOTE with repeat)
        indx = [random.randint(0, len(observations)-1) for _ in range(batch_size)]
        #next_indx = [i + 1 for i in indx]
        ob_batch = observations[indx] 
        ac_batch = actions[indx] 
        # clip self-generated actions
        ac_batch = np.clip(ac_batch, self.env.action_space.low, self.env.action_space.high)
        self.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))
        return ob_expert,  ac_expert, ob_batch, ac_batch 


    def train_onpolicy_discriminator(self, writer, logger, d_gradient_steps, d_learning_rate, batch_size, teacher_buffer, replay_buffer, seg, num_timesteps, sess, d_adam, nworkers):

        logger.log("Optimizing Discriminator...")
        #if step % 1000 == 0:
        logger.log(fmt_row(13, self.loss_name ))
        #timesteps_per_batch = len(observation)

        # NOTE: uses only the last g step for observation
        d_losses = []  # list of tuples, each of which gives the loss for a minibatch
        for i in range(d_gradient_steps):
            steps = num_timesteps + (i + 1) * (seg["total_timestep"] / d_gradient_steps)
            if replay_buffer is None:
                ob_expert, ac_expert, ob_batch, ac_batch = self.generate_onpolicy_data(teacher_buffer, seg, batch_size, sess)
            else:
                ob_expert, ac_expert, ob_batch, ac_batch = self.generate_offpolicy_data(teacher_buffer, replay_buffer, batch_size, sess, unscale=False)
                #ob_expert, ac_expert, ob_batch, ac_batch = self.generate_explore_data(teacher_buffer, replay_buffer, seg, batch_size, sess, unscale=False)
            gail_summary, *newlosses, grad = self.lossandgrad(ob_batch, ac_expert, ob_expert, ac_batch)
            d_adam.update(self.allmean(grad, nworkers), d_learning_rate)
            d_losses.append(newlosses)
            if writer is not None:
                writer.add_summary(gail_summary, steps)

            del ob_batch, ac_batch, ob_expert, ac_expert
        #if step % 1000 == 0:
        logger.log(fmt_row(13, np.mean(d_losses, axis=0)))


    def generate_explore_data(self, teacher_buffer, replay_buffer, segs, batch_size, sess, unscale=True):
        expert_batch = teacher_buffer.sample(batch_size)
        ob_expert, ac_expert = expert_batch[:2]

        batch = replay_buffer.sample(batch_size // 2)
        # sample offpolicy data
        offpolicy_ob_batch, offpolicy_ac_batch =  batch[:2]
        # sample onpoliy data (NOTE with repeat)
        observations, actions = segs["observations"], segs["actions"]
        indx = [random.randint(0, len(observations)-1) for _ in range(batch_size // 2)]
        onpolicy_ob_batch, onpolicy_ac_batch = observations[indx], actions[indx] 
        ob_batch = np.concatenate((onpolicy_ob_batch, offpolicy_ob_batch), 0)
        ac_batch = np.concatenate((onpolicy_ac_batch, offpolicy_ac_batch), 0) 

        # update running mean/std for discriminator
        if self.normalize:
            self.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0),sess=sess)

        # Reshape actions if needed when using discrete actions
        if isinstance(self.action_space, gym.spaces.Discrete):
            if len(ac_batch.shape) == 2:
                ac_batch = ac_batch[:, 0]
            if len(ac_expert.shape) == 2:
                ac_expert = ac_expert[:, 0]
        if unscale:
            ac_expert = unscale_action(self.action_space, ac_expert)
            ac_batch = unscale_action(self.action_space, ac_batch)
        return ob_expert, ac_expert, ob_batch, ac_batch


    def generate_offpolicy_data(self, teacher_buffer, replay_buffer, batch_size, sess, unscale=True):
        expert_batch = teacher_buffer.sample(batch_size)
        ob_expert, ac_expert = expert_batch[:2]
        batch_size = ob_expert.shape[0]
        batch = replay_buffer.sample(batch_size)
        ob_batch, ac_batch =  batch[:2]

        # update running mean/std for discriminator
        if self.normalize:
            self.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0),sess=sess)

        # Reshape actions if needed when using discrete actions
        if isinstance(self.action_space, gym.spaces.Discrete):
            if len(ac_batch.shape) == 2:
                ac_batch = ac_batch[:, 0]
            if len(ac_expert.shape) == 2:
                ac_expert = ac_expert[:, 0]
        if unscale:
            ac_expert = unscale_action(self.action_space, ac_expert)
            ac_batch = unscale_action(self.action_space, ac_batch)
        return ob_expert, ac_expert, ob_batch, ac_batch


    def train_discriminator(self, writer, logger, step, d_gradient_steps, d_learning_rate, teacher_buffer, replay_buffer, batch_size, sess):

        logger.log("Optimizing Discriminator...")
        if step % 1000 == 0:
            logger.log(fmt_row(13, self.loss_name + ['discriminator-total-loss'] ))
        #timesteps_per_batch = len(observation)

        # NOTE: uses only the last g step for observation
        d_losses = []  # list of tuples, each of which gives the loss for a minibatch
        for i in range(d_gradient_steps):
            ob_expert, ac_expert, ob_batch, ac_batch = self.generate_offpolicy_data(teacher_buffer, replay_buffer, batch_size, sess, unscale=False)
            feed_dict = {
                self.generator_obs_ph: ob_batch,
                self.generator_acs_ph: ac_batch,
                self.expert_obs_ph: ob_expert,
                self.expert_acs_ph: ac_expert,
                self.d_learning_rate_ph: d_learning_rate
            }

            if writer is not None:
                run_ops = [self.gail_summary] + self.sample_loss + self.losses + [self.total_loss, self.train_op]
                out = sess.run(run_ops, feed_dict)
                summary = out.pop(0)
                writer.add_summary(summary, step)
            else:
                run_ops = self.sample_loss + self.losses + [self.total_loss, self.train_op]
                out = sess.run(run_ops, feed_dict)

            losses = out[2:-1]
            d_losses.append(losses)
            del ob_batch, ac_batch, ob_expert, ac_expert
        if step % 1000 == 0:
            logger.log(fmt_row(13, np.mean(d_losses, axis=0)))



