"""
Reference: https://github.com/openai/imitation
I follow the architecture from the official repository
"""
import gym
import tensorflow as tf
import numpy as np

from stable_baselines.common.mpi_running_mean_std import RunningMeanStd
from stable_baselines.common import tf_util as tf_util
from stable_baselines.common.distributions import ProbabilityDistribution, ProbabilityDistributionType
from stable_baselines.a2c.utils import linear


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
    :return: (tf.Tensor) the bernoulli entropy
    """
    ent = (1. - tf.nn.sigmoid(logits)) * logits - logsigmoid(logits)
    return ent

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

        
class VAE(object):
    """
    Tensorflow version of VAE implemented in https://github.com/aviralkumar2907/BEAR/blob/647ac1308af9eac37691ee734bab1cb3733db530/algos.py#L212
    Input: observation, 
    """
    def __init__(self, observation_space, action_space, latent_dim, hidden_size, scope="vae", normalize=True):
        
        # TODO: support images properly (using a CNN)
        self.scope = scope
        self.observation_shape = observation_space.shape
        self.actions_shape = action_space.shape  
        self.observation_space=observation_space
        self.action_space=action_space
        if isinstance(action_space, gym.spaces.Box):
            # Continuous action space
            self.discrete_actions = False
            self.n_actions = action_space.shape[0]
        elif isinstance(action_space, gym.spaces.Discrete):
            self.n_actions = action_space.n
            self.discrete_actions = True
        else:
            raise ValueError('Action space not supported: {}'.format(action_space))
        if latent_dim is None:
            latent_dim = self.n_actions * 2
        self.latent_dim = latent_dim
        self.hidden_size= hidden_size
        self.normalize = normalize
        self.obs_rms = None
        # Placeholders
        self.obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape,
                                               name="observations_ph")
        self.acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                               name="actions_ph")
        self.next_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape,
                                               name="next_observations_ph")

        obs_fl = self.preprocess_obs(self.obs_ph, reuse=False)
        acs_fl = self.preprocess_act(self.acs_ph)
        next_obs_fl = self.preprocess_obs(self.next_obs_ph, reuse=True)

        generator_input = tf.concat([obs_fl, acs_fl], axis=1)  # concatenate the two input -> form a transition 
        with tf.variable_scope(self.scope):
            z, mean, std = self.encoder(generator_input, reuse=False)
            decoded_act = self.decoder(obs_fl, z, reuse=False)
            reconstruct_loss = tf.compat.v1.losses.mean_squared_error(decoded_act, acs_fl)

            self.multiple_decoded_output = self.decode_multiple(obs_fl) # For calculate MMD loss in gaussian kernel
            self.decoded_output = self.decoder(obs_fl, reuse=True) # For imitating action given observation s
            self.decoded_next_output = self.decoder(next_obs_fl, reuse=True) # For imitating action given observation s'

        self.reconstruct_loss = tf.reduce_mean(reconstruct_loss)
        self.KL_loss = -0.5 * tf.reduce_mean(1 + tf.log(tf.pow(std, 2)) - tf.pow(mean,2) - tf.pow(std,2))  
        self.total_loss = self.reconstruct_loss + 0.5 * self.KL_loss
        self.losses = [self.reconstruct_loss, self.KL_loss]
        self.loss_names = ["Reconstruct-loss", "KL-loss"]

    def encoder(self, generator_input, reuse=False):
        with tf.variable_scope('encoder'):
            if reuse:
                tf.get_variable_scope().reuse_variables()  # reuse variables
            z = tf.contrib.layers.fully_connected(generator_input, self.hidden_size, activation_fn=tf.nn.relu) #e1
            z = tf.contrib.layers.fully_connected(z, self.hidden_size, activation_fn=tf.nn.relu) # e2
            mean = tf.contrib.layers.fully_connected(z, self.latent_dim, activation_fn=tf.identity)  # mean
            # clipped for numerical stability 
            log_std = tf.contrib.layers.fully_connected(z, self.latent_dim)  
            log_std = tf.clip_by_value(log_std, -4, 15)
            std = tf.exp(log_std)
        
        # build p(z|s,a)
        shape = (tf.shape(std)[0], self.latent_dim)
        z = mean + std * tf.random.normal(shape,0,1)  
        return z, mean, std
         

    def decoder(self, obs_fl, z=None, reuse=False): 
        if z is None:
            #z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(device).clamp(-0.5, 0.5)
            shape = (tf.shape(obs_fl)[0], self.latent_dim)
            z = tf.clip_by_value(tf.random.normal(shape, 0, 1), -0.5, 0.5)
        with tf.variable_scope('decoder'): 
            if reuse:
                tf.get_variable_scope().reuse_variables()  # reuse variables
            decoder_input = tf.concat([obs_fl, z], axis=1)
            d1 = tf.contrib.layers.fully_connected(decoder_input, self.hidden_size, activation_fn=tf.nn.relu)
            d2 = tf.contrib.layers.fully_connected(d1, self.hidden_size, activation_fn=tf.nn.relu)
            d3 = tf.contrib.layers.fully_connected(d2, self.n_actions, activation_fn=tf.nn.tanh)
            # rescale action 
            output =  d3 * tf.math.abs(self.action_space.low)
        return output

    def get_decoded(self, obs, sess=None):
        if sess is None:
            sess = tf.get_default_session()
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)

        feed_dict = {self.obs_ph: obs}
        decoded = sess.run(self.decoded_output, feed_dict) 
        return decoded

    def decode_multiple(self, obs_fl, z=None, num_decode=10):
        """Decode 10 samples atleast"""
        if z is None:
            shape = (tf.shape(obs_fl)[0], num_decode, self.latent_dim)
            z = tf.clip_by_value(tf.random.normal(shape, 0, 1), -0.5, 0.5) 
            # z = B * N * d
        with tf.variable_scope('decoder'):
            tf.get_variable_scope().reuse_variables()  # reuse variables
            dup_obs = tf.tile( tf.expand_dims(obs_fl, 0), tf.constant([num_decode, 1, 1]))
            dup_obs = tf.transpose(dup_obs,perm=(1,0,2))  
            #state.unsqueeze(0).repeat(num_decode, 1, 1).permute(1, 0, 2),

            decoder_input = tf.concat([dup_obs, z], axis=2)
            d1 = tf.contrib.layers.fully_connected(decoder_input, self.hidden_size, activation_fn=tf.nn.relu)
            d2 = tf.contrib.layers.fully_connected(d1, self.hidden_size, activation_fn=tf.nn.relu)
            d3 = tf.contrib.layers.fully_connected(d2, self.n_actions, activation_fn=tf.nn.tanh)
            # rescale action 
        return d3 * tf.math.abs(self.action_space.low)  

    def preprocess_obs(self, obs_ph, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            if self.normalize:
                with tf.variable_scope("obfilter"):
                    self.obs_rms = RunningMeanStd(shape=self.observation_shape)
                obs = (obs_ph - self.obs_rms.mean) / self.obs_rms.std
            else:
                obs = obs_ph
        ob_fl = tf.contrib.layers.flatten(obs)
        return ob_fl

    def preprocess_act(self, acs_ph):
        if self.discrete_actions:
            one_hot_actions = tf.one_hot(acs_ph, self.n_actions)
            actions_ph = tf.cast(one_hot_actions, tf.float32)
        else:
            actions_ph = acs_ph
        act_fl = tf.contrib.layers.flatten(actions_ph) 
        return act_fl
        

    def get_trainable_variables(self):
        """
        Get all the trainable variables from the graph

        :return: ([tf.Tensor]) the variables
        """
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope) 
        return var_list

    
