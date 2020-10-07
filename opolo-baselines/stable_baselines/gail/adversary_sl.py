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

class DiagGaussian(ProbabilityDistribution):
    def __init__(self, flat):
        """
        Probability distributions from multivariate Gaussian input
        :param flat: ([float]) the multivariate Gaussian input data
        """
        self.flat = flat
        mean, logstd = tf.split(axis=len(flat.shape) - 1, num_or_size_splits=2, value=flat)
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)
        super(DiagGaussian, self).__init__()

    def flatparam(self):
        return self.flat

    def mode(self):
        # Bounds are taken into account outside this class (during training only)
        return self.mean

    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[-1], tf.float32) \
               + tf.reduce_sum(self.logstd, axis=-1)

    def kl(self, other):
        assert isinstance(other, DiagGaussian)
        return tf.reduce_sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) /
                             (2.0 * tf.square(other.std)) - 0.5, axis=-1)

    def entropy(self):
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

    def sample(self):
        # Bounds are taken into acount outside this class (during training only)
        # Otherwise, it changes the distribution and breaks PPO2 for instance
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean),
                                                       dtype=self.mean.dtype)

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new multivariate Gaussian input
        :param flat: ([float]) the multivariate Gaussian input data
        :return: (ProbabilityDistribution) the instance from the given multivariate Gaussian input data
        """
        return cls(flat)

class DiagGaussianType(ProbabilityDistributionType): 
    def __init__(self, size):
        """
        The probability distribution type for multivariate Gaussian input
        :param size: (int) the number of dimensions of the multivariate gaussian
        """
        self.size = size

    def probability_distribution_class(self):
        return DiagGaussian

    def proba_distribution_from_flat(self, flat):
        """
        returns the probability distribution from flat probabilities
        :param flat: ([float]) the flat probabilities
        :return: (ProbabilityDistribution) the instance of the ProbabilityDistribution associated
        """
        return self.probability_distribution_class()(flat)

    def proba_distribution_from_latent(self, pi_latent_vector, init_scale=1.0, init_bias=0.0):
        mean = linear(pi_latent_vector, 'pi', self.size, init_scale=init_scale, init_bias=init_bias)
        logstd = tf.get_variable(name='pi/logstd', shape=[1, self.size], initializer=tf.zeros_initializer())
        pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1) 
        return self.proba_distribution_from_flat(pdparam), mean 

    def param_shape(self):
        return [2 * self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.float32


#class BehaviorCloninigClassifier(object):
#    def __init__(self, observation_space, action_space, hidden_size, scope="behavior_clone", normalize=True):
#        """
#
#        :param observation_space: (gym.spaces) 
#        :param hidden_size: ([int]) the hidden dimension for the MLP
#        :param scope: (str) tensorflow variable scope
#        :param normalize: (bool) Whether to normalize the state or not
#        """
#        # TODO: support images properly (using a CNN)
#        self.scope = scope
#        self.observation_shape = observation_space.shape 
#        self.observation_space = observation_space 
#        self.action_space = action_space
#        self.actions_shape = action_space.shape
#        self.phi_size = 128 
#        self.forward_lambda = 0.2
#
#        self.hidden_size = hidden_size
#        self.normalize = normalize
#        self.obs_rms = None
#        #print(observation_space.dtype)
#
#        if isinstance(action_space, gym.spaces.Box):
#            # Continuous action space
#            self.discrete_actions = False
#            self.n_actions = action_space.shape[0]
#        elif isinstance(action_space, gym.spaces.Discrete):
#            self.n_actions = action_space.n
#            self.discrete_actions = True
#        else:
#            raise ValueError('Action space not supported: {}'.format(action_space))
#
#        # Placeholders
#        self.obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape,
#                                               name="observations_ph") 
#        self.acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
#                                               name="actions_ph")
#        obs, acs = self.flatten(self.obs_ph, self.acs_ph, reuse=False) 
#
#        
#        with tf.variable_scope(self.scope):
#            # Build random network ph
#            logprob, prob = self.build_distribution_net(obs, acs)
#            #neglogp = self.build_distribution_net(obs, acs)
#        self.reward_op = -tf.log(1 - prob + 1e-8)
#        self.total_loss = tf.reduce_mean(-logprob, name='neglogp_loss')
#        self.mean_prob = tf.reduce_mean(prob, name='mean-prob')
#        self.losses = [self.total_loss, self.mean_prob]
#        self.loss_names = ["Neg-Log-Likelihood", "Likelihood"]
#
#    def flatten(self, obs_ph, acs_ph, reuse=False):
#        with tf.variable_scope(self.scope):
#            if reuse:
#                tf.get_variable_scope().reuse_variables()
#
#            if self.normalize:
#                with tf.variable_scope("bc_obfilter"):
#                    self.obs_rms = RunningMeanStd(shape=self.observation_shape) 
#                obs = (obs_ph - self.obs_rms.mean) / self.obs_rms.std 
#            else:
#                obs = obs_ph 
#            if self.discrete_actions:
#                one_hot_actions = tf.one_hot(acs_ph, self.n_actions)
#                actions_ph = tf.cast(one_hot_actions, tf.float32)
#            else:
#                actions_ph = acs_ph
#
#            ob_fl = tf.contrib.layers.flatten(obs)
#            act_fl = tf.contrib.layers.flatten(actions_ph)
#            return ob_fl, act_fl 
#
#    def build_distribution_net(self, input_ob_ph, input_ac_ph):
#        assert isinstance(self.action_space, gym.spaces.Box)
#        with tf.variable_scope(self.scope):
#            p_h1 = tf.contrib.layers.fully_connected(input_ob_ph, self.hidden_size, activation_fn=tf.nn.tanh,)
#            p_h2 = tf.contrib.layers.fully_connected(p_h1, self.hidden_size, activation_fn=tf.nn.tanh)
#            pi_latent = tf.contrib.layers.fully_connected(p_h2, self.n_actions, activation_fn=tf.nn.tanh) 
#            self.proba_distribution_type = DiagGaussianType(self.n_actions)
#            self._prob_dist, self._policy = self.proba_distribution_type.proba_distribution_from_latent(pi_latent, init_scale=0.01)
#            self._policy_proba = [self._prob_dist.mean, self._prob_dist.logstd]
#
#            ## get loss: -log-probability
#            #n_elts = self.n_actions  
#            n_elts = tf.math.reduce_prod(tf.cast(tf.shape(self._prob_dist.mean)[1:], tf.float32))
#            log_normalizer = n_elts / 2 * tf.math.log(2 * np.pi) + 1/2 * tf.reduce_sum(self._prob_dist.logstd, axis=1)  
#
#            # logprob = -tf.reduce_sum( tf.square(input_ac_ph - self._prob_dist.mean) / (2 * self._prob_dist.std), axis=1 ) - log_normalizer
#            logprob = -tf.reduce_sum( tf.exp( tf.log(     tf.square(input_ac_ph - self._prob_dist.mean)+ 1e-8   ) - self._prob_dist.logstd), axis=1 ) /2 - log_normalizer  
#            prob = tf.math.exp(logprob)
#            return logprob, prob
#    
#    def get_reward(self, sess, observation, actions):
#        assert isinstance(self.action_space, gym.spaces.Box)
#        observation = observation.reshape((-1,) + self.observation_space.shape)
#        actions = actions.reshape((-1, ) + self.action_space.shape)
#        return sess.run(self.reward_op, {self.obs_ph: observation, self.acs_ph:actions})
#            
#
#    def action_probability(self, sess, observation, actions=None):
#        assert isinstance(self.action_space, gym.spaces.Box)
#        observation = observation.reshape((-1,) + self.observation_space.shape)
#        mean, logstd = self.proba_step(observation, sess)  
#        # mean, logstd
#        if actions is not None:
#            actions = actions.reshape((-1, ) + self.action_space.shape)
#            std = np.exp(logstd)
#
#            n_elts = np.prod(mean.shape[1:])  # first dimension is batch size
#            log_normalizer = n_elts/2 * np.log(2 * np.pi) + 1/2 * np.sum(logstd, axis=1)
#
#            # Diagonal Gaussian action probability, for every action
#            logprob = -np.sum(np.square(actions - mean) / (2 * std), axis=1) - log_normalizer
#            return logprob
#        else:
#            return mean, logstd 
#
#
#    def proba_step(self, obs, sess):
#        """
#        return mean and std of gaussian probaility
#        """
#        return sess.run(self._policy_proba, {self.obs_ph: obs})
#
#    
#
#    def get_trainable_variables(self):
#        """
#        Get all the trainable variables from the graph
#
#        :return: ([tf.Tensor]) the variables
#        """
#        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope) 
#        return var_list
#    
#    def get_bonus(self, obs, sess=None):
#        if sess is None:
#            sess = tf.get_default_session()
#        if len(obs.shape) == 1:
#            obs = np.expand_dims(obs, 0)
#
#        feed_dict = {self.obs_ph: obs}
#        reward = sess.run(self.total_loss, feed_dict) 
#        return reward



class RandomCuriosityClassifier(object):
    def __init__(self, observation_space, hidden_size, scope="random_curiosity", normalize=True):
        """
        Random Network Distillation. 
        Two networks: 
        - a random network (fixed) to generate a random state embedding
        - a target network (learned) to imitate the random network


        :param observation_space: (gym.spaces) 
        :param hidden_size: ([int]) the hidden dimension for the MLP
        :param scope: (str) tensorflow variable scope
        :param normalize: (bool) Whether to normalize the state or not
        """
        # TODO: support images properly (using a CNN)
        self.scope = scope
        self.observation_shape = observation_space.shape 
        self.phi_size = 128 
        self.forward_lambda = 0.2 

        self.hidden_size = hidden_size
        self.normalize = normalize
        self.obs_rms = None

        # Placeholders
        self.obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape,
                                               name="observations_ph") 
        obs = self.flatten(self.obs_ph, reuse=False) 

        
        with tf.variable_scope(self.scope):
            # Build random network ph
            phi = self.build_phi(obs,'random_net') 
            # Build target network ph
            phi_hat = self.build_phi(obs,'targe_net') 
             
        #self.item_loss = 0.5 * tf.reduce_sum(tf.square(tf.subtract(phi_hat, tf.stop_gradient(phi))),axis=0)
        self.item_loss = tf.nn.l2_loss(tf.subtract(phi_hat, tf.stop_gradient(phi)))
        self.total_loss = tf.reduce_mean(self.item_loss, name='fowardloss')
        self.losses = [self.total_loss]
        self.loss_names = ["dynamic_loss"]

    def flatten(self, obs_ph, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            if self.normalize:
                with tf.variable_scope("c1_obfilter"):
                    self.obs_rms = RunningMeanStd(shape=self.observation_shape) 
                obs = (obs_ph - self.obs_rms.mean) / self.obs_rms.std 
            else:
                obs = obs_ph 
 
            ob_fl = tf.contrib.layers.flatten(obs) 
            return ob_fl 

    def build_phi(self, input_ph, scope):
        with tf.variable_scope(scope): 
            p_h1 = tf.contrib.layers.fully_connected(input_ph, self.hidden_size, activation_fn=tf.nn.relu,)
            p_h2 = tf.contrib.layers.fully_connected(p_h1, self.hidden_size, activation_fn=tf.nn.relu)
            phi = tf.contrib.layers.fully_connected(p_h2, self.phi_size, activation_fn=tf.nn.relu) 
        return phi

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
        if len(next_obs.shape) == 1:
            next_obs = np.expand_dims(next_obs, 0)

        feed_dict = {self.obs_ph: next_obs}
        reward = sess.run(self.item_loss, feed_dict) 
        return reward
        


class TransitionCuriosityClassifier(object):
    def __init__(self, observation_space, action_space, hidden_size, scope="transition_curiosity", normalize=True):
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
        obs, act, next_obs = self.flatten(self.obs_ph, self.acs_ph, self.next_obs_ph, reuse=False)
        # Build phi 
        phi1 = self.build_phi(obs, reuse=False)
        phi2 = self.build_phi(next_obs, reuse=True)

        
        with tf.variable_scope(self.scope):
            # Build inverse model
            self.invloss = self.build_inverse_graph(phi1, phi2, act)

            # Build forward model
            self.forwardloss, self.item_loss = self.build_forward_graph(phi1, act, phi2)

        self.total_loss = self.forward_lambda * self.forwardloss + (1-self.forward_lambda)*self.invloss  
        self.losses = [self.invloss, self.forwardloss, self.total_loss]
        self.loss_names = ["invloss", "forwardloss", "total_loss"]

    def flatten(self, obs_ph, acs_ph, next_obs_ph, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            if self.normalize:
                with tf.variable_scope("c1_obfilter"):
                    self.obs_rms = RunningMeanStd(shape=self.observation_shape)
                with tf.variable_scope("c2_obfilter"):
                    self.next_obs_rms = RunningMeanStd(shape=self.observation_shape)
                obs = (obs_ph - self.obs_rms.mean) / self.obs_rms.std
                next_obs = (next_obs_ph - self.next_obs_rms.mean) / self.next_obs_rms.std
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
            #p_h2 = tf.contrib.layers.fully_connected(p_h1, self.hidden_size)#, activation_fn=tf.nn.tanh)
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
            #item_loss = 0.5 * tf.reduce_sum(tf.square(tf.subtract(phi2_est, phi2)),axis=0)
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
        

class DiscriminatorCalssifier(object):
    def __init__(self, observation_space, action_space, hidden_size,
                 entcoeff=0.001, gradcoeff=0.001,scope="adversary", normalize=True):
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
        self.scope = scope
        self.observation_shape = observation_space.shape
        self.actions_shape = action_space.shape
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
        self.generator_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape,
                                               name="gail_observations_ph")
        self.generator_acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                               name="gail_actions_ph")
        self.expert_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape,
                                            name="gail_expert_observations_ph")
        self.expert_acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                            name="gail_expert_actions_ph")
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
        self.reward_op = -tf.log(1 - tf.nn.sigmoid(generator_logits) + 1e-8)
        self.confidence_op = tf.nn.sigmoid(generator_logits) 
        var_list = self.get_trainable_variables()
        self.lossandgrad = tf_util.function(
            [self.generator_obs_ph, self.generator_acs_ph, self.expert_obs_ph, self.expert_acs_ph],
            self.losses + [tf_util.flatgrad(self.total_loss, var_list)])

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


    def get_reward(self, obs, actions, sess=None):
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

class TransitionClassifier(object):
    def __init__(self, observation_space, action_space, hidden_size,
                 entcoeff=0.001, scope="adversary", normalize=True):
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
        self.scope = scope
        self.observation_shape = observation_space.shape
        self.actions_shape = action_space.shape
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
        self.generator_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape,
                                               name="observations_ph")
        self.generator_acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                               name="actions_ph")
        self.expert_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape,
                                            name="expert_observations_ph")
        self.expert_acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                            name="expert_actions_ph")
        # Build graph
        generator_logits = self.build_graph_flatten(self.generator_obs_ph, self.generator_acs_ph, reuse=False)
        expert_logits = self.build_graph_flatten(self.expert_obs_ph, self.expert_acs_ph, reuse=True)
        # Build accuracy
        generator_acc = tf.reduce_mean(tf.cast(tf.nn.sigmoid(generator_logits) < 0.5, tf.float32))
        expert_acc = tf.reduce_mean(tf.cast(tf.nn.sigmoid(expert_logits) > 0.5, tf.float32))
        # Build regression loss
        # let x = logits, z = targets.
        # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits,
                                                                 labels=tf.zeros_like(generator_logits))
        generator_loss = tf.reduce_mean(generator_loss)
        expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits, labels=tf.ones_like(expert_logits))
        expert_loss = tf.reduce_mean(expert_loss)
        # Build entropy loss
        logits = tf.concat([generator_logits, expert_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        entropy_loss = -entcoeff * entropy

        # Loss + Accuracy terms
        self.losses = [generator_loss, expert_loss, entropy, entropy_loss, generator_acc, expert_acc]
        self.loss_name = ["generator_loss", "expert_loss", "entropy", "entropy_loss", "generator_acc", "expert_acc"]
        self.total_loss = generator_loss + expert_loss + entropy_loss
        # Build Reward for policy
        self.reward_op = -tf.log(1 - tf.nn.sigmoid(generator_logits) + 1e-8)
        self.confidence_op = tf.nn.sigmoid(generator_logits) 
        var_list = self.get_trainable_variables()
        self.lossandgrad = tf_util.function(
            [self.generator_obs_ph, self.generator_acs_ph, self.expert_obs_ph, self.expert_acs_ph],
            self.losses + [tf_util.flatgrad(self.total_loss, var_list)])

    def build_graph_flatten(self, obs_ph, acs_ph, reuse=False):
        """
        build the graph, where observations and actions are all flattened,
        in order to address image observations.

        :param obs_ph: (tf.Tensor) the observation placeholder
        :param acs_ph: (tf.Tensor) the action placeholder
        :param reuse: (bool)
        :return: (tf.Tensor) the graph output
        """

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
            _input = tf.concat([ob_fl, act_fl], axis=1)  # concatenate the two input -> form a transition
            p_h1 = tf.contrib.layers.fully_connected(_input, self.hidden_size, activation_fn=tf.nn.tanh)
            p_h2 = tf.contrib.layers.fully_connected(p_h1, self.hidden_size, activation_fn=tf.nn.tanh)
            logits = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.identity)
        return logits

    def build_graph(self, obs_ph, acs_ph, reuse=False):
        """
        build the graph

        :param obs_ph: (tf.Tensor) the observation placeholder
        :param acs_ph: (tf.Tensor) the action placeholder
        :param reuse: (bool)
        :return: (tf.Tensor) the graph output
        """
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

            _input = tf.concat([obs, actions_ph], axis=1)  # concatenate the two input -> form a transition
            p_h1 = tf.contrib.layers.fully_connected(_input, self.hidden_size, activation_fn=tf.nn.tanh)
            p_h2 = tf.contrib.layers.fully_connected(p_h1, self.hidden_size, activation_fn=tf.nn.tanh)
            logits = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.identity)
        return logits

    def get_trainable_variables(self):
        """
        Get all the trainable variables from the graph

        :return: ([tf.Tensor]) the variables
        """
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_reward(self, obs, actions):
        """
        Predict the reward using the observation and action

        :param obs: (tf.Tensor or np.ndarray) the observation
        :param actions: (tf.Tensor or np.ndarray) the action
        :return: (np.ndarray) the reward
        """
        sess = tf.get_default_session()
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, 0)
        elif len(actions.shape) == 0:
            # one discrete action
            actions = np.expand_dims(actions, 0)

        feed_dict = {self.generator_obs_ph: obs, self.generator_acs_ph: actions}
        reward = sess.run(self.reward_op, feed_dict)
        return reward

    def get_confidence(self, obs, actions):
        """
        Predict the reward using the observation and action

        :param obs: (tf.Tensor or np.ndarray) the observation
        :param actions: (tf.Tensor or np.ndarray) the action
        :return: (np.ndarray) the reward
        """
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