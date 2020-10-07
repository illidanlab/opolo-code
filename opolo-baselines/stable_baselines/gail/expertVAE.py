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
    for state transition model: Input: s, output s' (based on expert dataset) 
    for inverse dynamic model: Input s,s', output a (based on self-generated dataset)
    """
    def __init__(self, observation_space, action_space, latent_dim, hidden_size, scope="vae", reward_coeff=0.1, normalize=True):
        
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
            latent_dim =  128
        self.latent_dim = latent_dim
        self.hidden_size= hidden_size
        self.normalize = normalize
        self.obs_rms = None
        self.reward_coeff = reward_coeff
        # Placeholders
        self.obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape,
                                               name="observations_ph")
        self.acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                               name="actions_ph")
        self.next_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape,
                                               name="next_observations_ph")
        self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

        self.pi_acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                               name="pi_actions_ph")

        obs_fl = self.preprocess_obs(self.obs_ph, reuse=False)
        next_obs_fl = self.preprocess_obs(self.next_obs_ph, reuse=True)
        acs_fl = self.preprocess_act(self.acs_ph)
        pi_acs_fl = self.preprocess_act(self.pi_acs_ph)
        transition_input = tf.concat([obs_fl, next_obs_fl], axis=1)  # concatenate s and s' -> form a transition 
        # build f(s) -> s'
        with tf.variable_scope(self.scope):
            z, mean, std = self.encoder(obs_fl, scope='state_encoder', reuse=False)
            decoded_next_obs = self.state_decoder(obs_fl, z, scope='state_decoder', reuse=False)
            reconstruct_s_loss = tf.compat.v1.losses.mean_squared_error(decoded_next_obs, next_obs_fl)

        # loss for training f(s)
        self.reconstruct_s_loss = tf.reduce_mean(reconstruct_s_loss)
        self.s_KL_loss = -0.5 * tf.reduce_mean(1 + tf.log(tf.pow(std, 2)) - tf.pow(mean,2) - tf.pow(std,2))  
        self.s_total_loss = self.reconstruct_s_loss + 0.5 * self.s_KL_loss
        self.s_losses = [self.reconstruct_s_loss, self.s_KL_loss]
        self.s_loss_names = ["Reconstruct-state-loss", "state-KL-loss"]

        #self.lossandgrad = tf_util.function(
        #    [self.generator_obs_ph, self.generator_acs_ph, self.expert_obs_ph, self.expert_acs_ph],
        #    [self.gail_summary] + self.losses + [tf_util.flatgrad(self.total_loss, var_list)])

        # build M(s,s') -> a
        with tf.variable_scope(self.scope): 
            t_z, _, _ = self.encoder(transition_input, scope='action_encoder', reuse=False)
            decoded_acs = self.action_decoder(transition_input, t_z, scope='action_decoder', reuse=False)
            reconstruct_a_loss = tf.compat.v1.losses.mean_squared_error(decoded_acs, acs_fl)
            #self.decoded_acs = self.state_decoder(obs_fl, scope='state_decoder', reuse=True) # For imitating action given observation s

        # loss for training M(s,s')
        self.reconstruct_a_loss = tf.reduce_mean(reconstruct_a_loss)
        self.a_KL_loss = -0.5 * tf.reduce_mean(1 + tf.log(tf.pow(std, 2)) - tf.pow(mean,2) - tf.pow(std,2))  
        self.a_total_loss = self.reconstruct_a_loss + 0.5 * self.a_KL_loss
        self.a_losses = [self.reconstruct_a_loss, self.a_KL_loss]
        self.a_loss_names = ["Reconstruct-action-loss", "action-KL-loss"]

        var_list=self.get_trainable_variables()
        state_vae_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
        action_vae_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
        self.train_state_vae_op=state_vae_optimizer.minimize(self.s_total_loss, var_list=var_list)
        self.train_action_vae_op=action_vae_optimizer.minimize(self.a_total_loss, var_list=var_list)
        # Log discriminator scalars for debugging purposes
        state_vae_summaries = []
        for i, loss_name in enumerate(self.s_loss_names):
            i = tf.summary.scalar(loss_name, self.s_losses[i])
            state_vae_summaries.append(i)
        self.state_vae_summary = tf.summary.merge(state_vae_summaries)

        action_vae_summaries = []
        for i, loss_name in enumerate(self.a_loss_names):
            i = tf.summary.scalar(loss_name, self.a_losses[i])
            action_vae_summaries.append(i)
        self.action_vae_summary = tf.summary.merge(action_vae_summaries)
    
        # build operator to get a_E given s_\pi
        # first use f(s) to infer s'
        # then use M(s, f(s)) to infer a
        # then get log(a|s) (optional)
        with tf.variable_scope(self.scope): 
            decoded_next_obs = self.state_decoder(obs_fl, scope='state_decoder', reuse=True) # s' = f(s) 
            forward_input = tf.concat([obs_fl, decoded_next_obs], axis=1) # input = (s, f(s)) 
            inverse_input = tf.concat([obs_fl, next_obs_fl], axis=1)      # input = (s,s')
            self.inferred_forward_acs = self.action_decoder(forward_input, scope='action_decoder', reuse=True) # M(s,f(s))  
            self.inferred_inverse_acs = self.action_decoder(inverse_input, scope='action_decoder', reuse=True)
            
            #self.pi_acs_reward = - tf.compat.v1.losses.mean_squared_error(self.inferred_forward_acs, pi_acs_fl)
            # use huber-loss to avoid outliers
            self.pi_acs_reward = - self.reward_coeff * tf.compat.v1.losses.huber_loss(self.inferred_forward_acs, pi_acs_fl, delta=0.5)
        


    def encoder(self, generator_input, scope=None, reuse=False):
        if not scope:
            scope = 'encoder'
        with tf.variable_scope(scope):
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
         

    def state_decoder(self, obs_fl, z=None, scope=None, reuse=False): 
        if z is None:
            #z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(device).clamp(-0.5, 0.5)
            shape = (tf.shape(obs_fl)[0], self.latent_dim)
            z = tf.clip_by_value(tf.random.normal(shape, 0, 1), -0.5, 0.5)
        if not scope:
            scope = 'decoder'
        with tf.variable_scope(scope): 
            if reuse:
                tf.get_variable_scope().reuse_variables()  # reuse variables
            decoder_input = tf.concat([obs_fl, z], axis=1)
            d1 = tf.contrib.layers.fully_connected(decoder_input, self.hidden_size, activation_fn=tf.nn.relu)
            d2 = tf.contrib.layers.fully_connected(d1, self.hidden_size, activation_fn=tf.nn.relu)
            output = tf.contrib.layers.fully_connected(d2, self.observation_shape[0], activation_fn=tf.nn.tanh)
        return output


    def action_decoder(self, state_input, z=None, scope=None, reuse=False): 
        if z is None:
            #z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(device).clamp(-0.5, 0.5)
            shape = (tf.shape(state_input)[0], self.latent_dim)
            z = tf.clip_by_value(tf.random.normal(shape, 0, 1), -0.5, 0.5)
        if not scope:
            scope = 'decoder'
        with tf.variable_scope(scope): 
            if reuse:
                tf.get_variable_scope().reuse_variables()  # reuse variables
            decoder_input = tf.concat([state_input, z], axis=1)
            d1 = tf.contrib.layers.fully_connected(decoder_input, self.hidden_size, activation_fn=tf.nn.relu)
            d2 = tf.contrib.layers.fully_connected(d1, self.hidden_size, activation_fn=tf.nn.relu)
            output = tf.contrib.layers.fully_connected(d2, self.n_actions, activation_fn=tf.nn.tanh)
        return output


    def get_inverse_action(self, obs, next_obs, sess=None): 
        # input s, s' \sim Teacher, get a_E
        if sess is None:
            sess = tf.get_default_session()
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)

        feed_dict = {self.obs_ph: obs, self.next_obs_ph: next_obs}
        
        decoded = sess.run(self.inferred_inverse_acs, feed_dict) 
        return decoded
    
    def get_forward_action(self, obs, sess=None): 
        # input s \sim \pi, get a_E
        if sess is None:
            sess = tf.get_default_session()
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)

        feed_dict = {self.obs_ph: obs}
        decoded = sess.run(self.inferred_forward_acs, feed_dict) 
        return decoded

    def get_reward(self, obs, acs, sess=None):
        # input (s,a) , get |a - M(s, f(s))|^2
        if sess is None:
            sess = tf.get_default_session()
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)

        feed_dict = {self.obs_ph: obs, self.pi_acs_ph: acs}
        acs_loss = sess.run(self.pi_acs_reward, feed_dict) 
        return acs_loss

        
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
        

    def get_trainable_variables(self, scope=None):
        """
        Get all the trainable variables from the graph

        :return: ([tf.Tensor]) the variables
        """
        if scope is None:
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope) 
        else:
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope + '/' + scope) 

    
    def generate_state_data(self, teacher_buffer, batch_size, sess):
        expert_batch = teacher_buffer.sample(batch_size)
        ob_expert, _, _, next_ob_expert = expert_batch[:4]
        # update running mean/std for discriminator
        if self.normalize:
            self.obs_rms.update(ob_expert ,sess=sess)
        return ob_expert, next_ob_expert

    def train_state_vae(self, writer, logger, current_lr, step, train_steps, teacher_buffer, batch_size, sess):
        logger.log("Optimizing State Transition VAE ...")
        #logger.log(fmt_row(13, self.bc.loss_names ))
        vae_losses = []  # list of tuples, each of which gives the loss for a minibatch
        for i in range(int(train_steps)):
            ob_expert, next_ob_expert = self.generate_state_data(teacher_buffer, batch_size, sess)
            feed_dict = {
                self.obs_ph: ob_expert,
                #self.acs_ph: ac_batch,
                self.next_obs_ph: next_ob_expert,
                self.learning_rate_ph: current_lr
            }

            if writer is not None:
                run_ops = [self.state_vae_summary] + [self.train_state_vae_op] + self.s_losses
                out = sess.run(run_ops, feed_dict)
                summary = out.pop(0)
                writer.add_summary(summary, step)
            else:
                run_ops = [self.train_state_vae_op] + self.s_losses
                out = sess.run(run_ops, feed_dict)
            #### 昏哥线 ###
            loss_dim = len(self.s_losses)
            losses = out[-loss_dim:]
            vae_losses.append(losses)
            del ob_expert, next_ob_expert
            if i % 100 == 0:
                logger.log(fmt_row(13, np.mean(vae_losses, axis=0)))

    def generate_action_data(self, replay_buffer, batch_size, sess):
        batch = replay_buffer.sample(batch_size)
        ob_batch, ac_batch, _, next_ob_batch = batch[:4]
        # update running mean/std for discriminator
        return ob_batch, ac_batch, next_ob_batch 


    def train_action_vae(self, writer, logger, current_lr, step, train_steps, replay_buffer, batch_size, sess):
        logger.log("Optimizing Action Transition VAE ...")
        vae_losses = []  # list of tuples, each of which gives the loss for a minibatch
        for i in range(int(train_steps)):
            ob_batch, ac_batch, next_ob_batch = self.generate_action_data(replay_buffer, batch_size, sess)
            feed_dict = {
                self.obs_ph: ob_batch,
                self.acs_ph: ac_batch,
                self.next_obs_ph: next_ob_batch,
                self.learning_rate_ph: current_lr
            }

            if writer is not None:
                run_ops = [self.action_vae_summary] + [self.train_action_vae_op] + self.a_losses
                out = sess.run(run_ops, feed_dict)
                summary = out.pop(0)
                writer.add_summary(summary, step)
            else:
                run_ops = [self.train_action_vae_op] + self.a_losses
                out = sess.run(run_ops, feed_dict)
            #### 昏哥线 ###
            loss_dim = len(self.a_losses)
            losses = out[-loss_dim:]
            vae_losses.append(losses)
            del ob_batch, ac_batch, next_ob_batch
            if i % 100 == 0:
                logger.log(fmt_row(13, np.mean(vae_losses, axis=0)))