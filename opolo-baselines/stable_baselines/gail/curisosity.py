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
    

class TransitionCuriosityClassifier(object):
    def __init__(self, observation_space, action_space, hidden_size=256, scope="transition_curiosity", normalize=True):
        """
        Reward regression from observations and transitions

        :param observation_space: (gym.spaces)
        :param action_space: (gym.spaces)
        :param hidden_size: ([int]) the hidden dimension for the MLP
        :param scope: (str) tensorflow variable scope
        :param normalize: (bool) Whether to normalize the reward or not
        """
        # TODO: support images properly (using a CNN)
        self.scope = scope
        self.observation_shape = observation_space.shape
        self.action_space = action_space
        self.actions_shape = action_space.shape
        self.phi_size = 128 
        self.forward_lambda = 0.2
        #print(observation_space.dtype)

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
        self.d_learning_rate_ph = tf.placeholder(tf.float32, [], name="d_learning_rate_ph")
        obs, act, next_obs = self.flatten(self.obs_ph, self.acs_ph, self.next_obs_ph, reuse=False)
        # Build phi 
        phi1 = self.build_phi(obs, reuse=False)
        phi2 = self.build_phi(next_obs, reuse=True)

        
        with tf.variable_scope(self.scope):
            # Build inverse model
            self.invloss = self.build_inverse_graph(phi1, phi2, act)

            # Build forward model
            self.forwardloss, self.item_loss = self.build_forward_graph(phi1, act, phi2)
            
        var_list = self.get_trainable_variables()

        self.total_loss = self.forward_lambda * self.forwardloss + (1-self.forward_lambda)*self.invloss  
        self.losses = [self.invloss, self.forwardloss, self.total_loss]
        self.loss_names = ["invloss", "forwardloss", "total_loss"]
        self.lossandgrad = tf_util.function(
            [self.obs_ph, self.acs_ph, self.next_obs_ph], self.losses + [tf_util.flatgrad(self.total_loss, var_list)])
        train_optimizer = tf.train.AdamOptimizer(learning_rate=self.d_learning_rate_ph)
        self.train_op=train_optimizer.minimize(self.total_loss, var_list=var_list)

    def flatten(self, obs_ph, acs_ph, next_obs_ph, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            if self.normalize:
                with tf.variable_scope("c1_obfilter"):
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
            act_fl = tf.contrib.layers.flatten(actions_ph)
            next_ob_fl = tf.contrib.layers.flatten(next_obs)
            return ob_fl, act_fl, next_ob_fl

    def build_phi(self, input_ph, reuse=True):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            p_h1 = tf.contrib.layers.fully_connected(input_ph, self.hidden_size)#, activation_fn=tf.nn.tanh)
            phi = tf.contrib.layers.fully_connected(p_h1, self.phi_size, activation_fn=tf.nn.relu) 
        return phi

    def build_inverse_graph(self, phi1, phi2, act): 
        with tf.variable_scope('inverse'):
            g = tf.concat([phi1, phi2], axis=1)
            ac_logit = tf.contrib.layers.fully_connected(g, self.n_actions)
            invloss = tf.nn.l2_loss(ac_logit - act)
            invloss = tf.reduce_mean(invloss, name="invloss")
        return invloss

    def build_forward_graph(self, phi1, act, phi2):
        with tf.variable_scope('forward'): 
            state_size = self.phi_size 
            f = tf.concat([phi1, act], axis=1)
            f = tf.contrib.layers.fully_connected(f, self.hidden_size, activation_fn=tf.nn.relu)
            f = tf.contrib.layers.fully_connected(f, self.hidden_size, activation_fn=tf.nn.relu)
            phi2_est = tf.contrib.layers.fully_connected(f, state_size) 
            item_loss = tf.nn.l2_loss(tf.subtract(phi2_est, phi2))
            forwardloss = tf.reduce_mean(item_loss, name='fowardloss')
        return forwardloss, item_loss

    def get_trainable_variables(self):
        """
        Get all the trainable variables from the graph

        :return: ([tf.Tensor]) the variables
        """
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope) 
        return var_list
    
    def get_bonus(self, obs, act, next_obs, sess=None):
        if sess is None:
            sess = tf.get_default_session()
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        if len(act.shape) == 1:
            act = np.expand_dims(act, 0)
        elif len(act.shape) == 0:
            # one discrete action
            act = np.expand_dims(act, 0)

        feed_dict = {self.obs_ph: obs, self.acs_ph: act, self.next_obs_ph: next_obs}
        #reward = sess.run(self.forwardloss, feed_dict) 
        reward = sess.run(self.item_loss, feed_dict) 
        return reward 

    def get_reward(self, obs, actions, next_obs, unscale=True, sess=None):
        """
        Predict the reward using the observations and action
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

        if unscale:
            actions = unscale_action(self.action_space, actions)
        feed_dict = {self.obs_ph: obs, self.acs_ph: actions, self.next_obs_ph: next_obs}
        reward = sess.run(self.item_loss, feed_dict) 
        return reward 


    def train_classifier(self, logger, step, d_gradient_steps, d_learning_rate, replay_buffer, batch_size, sess):

        logger.log("Optimizing Curiosity Explorer...")
        if step % 1000 == 0:
            logger.log(fmt_row(13, self.loss_names))
        #timesteps_per_batch = len(observation)

        # NOTE: uses only the last g step for observation
        d_losses = []  # list of tuples, each of which gives the loss for a minibatch
        for i in range(d_gradient_steps):

            batch = replay_buffer.sample(batch_size)
            ob_batch, ac_batch, _, next_ob_batch =  batch[:4]
            feed_dict = {
                self.obs_ph: ob_batch,
                self.acs_ph: ac_batch,
                self.next_obs_ph: next_ob_batch,
                self.d_learning_rate_ph: d_learning_rate
            }

            run_ops = self.losses + [self.total_loss, self.train_op]
            out = sess.run(run_ops, feed_dict)

            losses = out[2:-1]
            d_losses.append(losses)
            del ob_batch, ac_batch, next_ob_batch
        if step % 1000 == 0:
            logger.log(fmt_row(13, np.mean(d_losses, axis=0)))