"""
Reference: https://github.com/openai/imitation
I follow the architecture from the official repository
"""
import gym
import tensorflow as tf
import numpy as np

from stable_baselines.common.mpi_running_mean_std import RunningMeanStd
from stable_baselines.common import fmt_row


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


class InverseModel(object):
    def __init__(self, observation_space, action_space, hidden_size=256,
                 reg_coeff=0.001, scope="inverse", learning_rate=3e-4, normalize=True):
        """
        Reward regression from observations and transitions

        :param observation_space: (gym.spaces)
        :param action_space: (gym.spaces)
        :param hidden_size: ([int]) the hidden dimension for the MLP
        :param entcoeff: (float) the entropy loss weight
        :param scope: (str) tensorflow variable scope
        :param normalize: (bool) Whether to normalize the reward or not
        """
        # TODO: support images properly (using CNN)
        self.scope = scope
        self.observation_shape = observation_space.shape
        self.actions_shape = action_space.shape
        self.action_space = action_space
        self.observation_space = observation_space
        self.d_learning_rate = learning_rate
        self.reg_coeff = reg_coeff

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
        self.obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape,
                                               name="observations_ph")
        self.acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                               name="actions_ph")
        self.next_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape,
                                               name="next_observations_ph")
        # Build graph
        states, action = self.flatten(self.obs_ph, self.next_obs_ph, self.acs_ph, reuse=False)
        self.inverse_actions = self.build_IDM_graph(states, reuse=False)

        var_list = self.get_trainable_variables()
 
        # Build predction loss     
        # NOTE TBD: loss for discrete action space
        self.prediction_loss = tf.reduce_mean(tf.square(self.inverse_actions - action)) 
        # Build regularization loss
        regularization_loss = self.reg_coeff * tf.add_n([tf.nn.l2_loss(var) for var in var_list])

        # Loss + Accuracy terms
        self.losses = [self.prediction_loss, regularization_loss]
        self.loss_name = ["prediction_loss", "regularization_loss"]
        self.total_loss = self.prediction_loss + regularization_loss

        idm_optimizer = tf.train.AdamOptimizer(learning_rate=self.d_learning_rate)
        self.train_op=idm_optimizer.minimize(self.total_loss, var_list=var_list)
        # Log discriminator scalars for debugging purposes
        idm_scalar_summaries = []
        for i, loss_name in enumerate(self.loss_name):
            i = tf.summary.scalar(loss_name, self.losses[i])
            idm_scalar_summaries.append(i)
        self.idm_summary = tf.summary.merge(idm_scalar_summaries)


    def flatten(self, obs_ph, next_obs_ph, acs_ph, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            if self.normalize:
                with tf.variable_scope("obfilter"):
                    self.obs_rms = RunningMeanStd(shape=self.observation_shape)
                obs = (obs_ph - self.obs_rms.mean) / self.obs_rms.std
                next_obs = (next_obs_ph - self.obs_rms.mean) / self.obs_rms.std
            else:
                obs = obs_ph
                next_obs = next_obs_ph

            if self.discrete_actions:
                one_hot_actions = tf.one_hot(acs_ph, self.n_actions)
                actions_ph = tf.cast(one_hot_actions, tf.float32)
            else:
                actions_ph = acs_ph

            ob_fl = tf.contrib.layers.flatten(obs)
            next_ob_fl = tf.contrib.layers.flatten(next_obs)
            act_fl = tf.contrib.layers.flatten(actions_ph)
            flatten_states = tf.concat([ob_fl, next_ob_fl], axis=1)  # concatenate the two input -> form a transition
            return flatten_states, act_fl

    def build_IDM_graph(self, states, reuse=True):
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
            p_h1 = tf.contrib.layers.fully_connected(states, self.hidden_size, activation_fn=tf.nn.tanh)
            p_h2 = tf.contrib.layers.fully_connected(p_h1, self.hidden_size, activation_fn=tf.nn.tanh)
            logits = tf.contrib.layers.fully_connected(p_h2, self.n_actions, activation_fn=tf.identity) 
            # NOTE should we clip out put logits ?
        return logits 


    def get_trainable_variables(self):
        """
        Get all the trainable variables from the graph

        :return: ([tf.Tensor]) the variables
        """
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)



    def generate_idm_data(self, replay_buffer, batch_size, sess):
        batch_data = replay_buffer.sample(batch_size)
        ob_batch, ac_batch, next_ob_batch = batch_data[0], batch_data[1], batch_data[3]

        # update running mean/std for discriminator
        if self.normalize:
            self.obs_rms.update(np.concatenate((ob_batch, next_ob_batch), 0),sess=sess)

        # Reshape actions if needed when using discrete actions
        if isinstance(self.action_space, gym.spaces.Discrete):
            if len(ac_batch.shape) == 2:
                ac_batch = ac_batch[:, 0]
            if len(ac_batch.shape) == 2:
                ac_batch = ac_batch[:, 0]

        return ob_batch, ac_batch, next_ob_batch


    def get_inverse_action(self, obs, next_obs, sess):
        feed_dict = {self.obs_ph: obs, self.next_obs_ph: next_obs}
        inverse_actions = sess.run([self.inverse_actions], feed_dict)
        return inverse_actions

    def get_inverse_loss(self, obs, actions, next_obs, sess):
        feed_dict = {self.obs_ph: obs, self.acs_ph: actions, self.next_obs_ph: next_obs}
        loss = sess.run([self.prediction_loss], feed_dict)
        return loss

    def train_idm(self, writer, logger, step, d_gradient_steps, replay_buffer, batch_size, sess):

        logger.log("Optimizing Inverse Model...")
        if step % 1000 == 0:
            logger.log(fmt_row(13, self.loss_name + ['idm-total-loss'] ))
        #timesteps_per_batch = len(observation)

        # NOTE: uses only the last g step for observation
        d_losses = []  # list of tuples, each of which gives the loss for a minibatch
        for i in range(d_gradient_steps):
            ob_batch, ac_batch, next_ob_batch = self.generate_idm_data(replay_buffer, batch_size, sess)
            feed_dict = {
                self.obs_ph: ob_batch,
                self.acs_ph: ac_batch,
                self.next_obs_ph: next_ob_batch,
            }

            if writer is not None:
                run_ops = [self.idm_summary] + self.losses + [self.total_loss, self.train_op]
                out = sess.run(run_ops, feed_dict)
                summary = out.pop(0)
                writer.add_summary(summary, step)
            else:
                run_ops = self.losses + [self.total_loss, self.train_op]
                out = sess.run(run_ops, feed_dict)

            losses = out[:-1]
            d_losses.append(losses)
            del ob_batch, ac_batch, next_ob_batch
        if step % 500 == 0:
            logger.log(fmt_row(13, np.mean(d_losses, axis=0)))



