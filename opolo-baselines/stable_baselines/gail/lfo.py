"""
Reference: https://github.com/openai/imitation
I follow the architecture from the official repository
"""
import gym
import tensorflow as tf
import numpy as np

from stable_baselines.common.mpi_running_mean_std import RunningMeanStd
from stable_baselines.common import tf_util as tf_util
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


class WGANDiscriminator(object):
    def __init__(self, observation_space, hidden_size=256,
                 entcoeff=0.001, gradcoeff=10,scope="wasserstein", learning_rate=3e-4, normalize=True):
        """
        Reward regression from observations and transitions

        :param observation_space: (gym.spaces)
        :param hidden_size: ([int]) the hidden dimension for the MLP
        :param entcoeff: (float) the entropy loss weight
        :param scope: (str) tensorflow variable scope
        :param normalize: (bool) Whether to normalize the reward or not
        """
        # TODO: support images properly (using a CNN)
        self.scope = scope
        self.observation_shape = observation_space.shape
        self.observation_space = observation_space
        self.d_learning_rate = learning_rate

        self.hidden_size = hidden_size
        self.normalize = normalize
        self.obs_rms = None

        # Placeholders
        self.generator_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape,
                                               name="w1_observations_ph")
        self.expert_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape,
                                            name="w1_expert_observations_ph")
        self.generator_next_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape,
                                               name="w1_next_observations_ph")
        self.expert_next_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape,
                                            name="w1_expert_next_observations_ph")
        # Build graph
        generator_input = self.flatten(self.generator_obs_ph, self.generator_next_obs_ph, reuse=False)
        generator_input = tf.stop_gradient(generator_input)
        generator_logits = self.build_WS_graph(generator_input, reuse=False)

        expert_input = self.flatten(self.expert_obs_ph, self.expert_next_obs_ph, reuse=True)
        expert_input = tf.stop_gradient(expert_input)
        expert_logits = self.build_WS_graph(expert_input, reuse=True)


        # Build loss function
        generator_out_mean = tf.reduce_mean(generator_logits) 
        expert_out_mean = tf.reduce_mean(expert_logits)

        # Build Gradient-penalty loss
        alpha_shape = self.observation_shape[0] * 2 
        alpha = np.random.uniform(size=(1, alpha_shape))
        inter = alpha * generator_input + (1 - alpha) * expert_input 
        with tf.GradientTape() as tape:
            tape.watch(inter)
            inter_output = self.build_WS_graph(inter, reuse=True)
            grad = tape.gradient(inter_output, [inter])[0]
            grad = tf.pow(tf.norm(grad, axis=-1) - 1, 2)
            grad = tf.reduce_mean(grad)
        grad_penalty =  gradcoeff * grad

        # Loss + Accuracy terms
        self.total_loss = generator_out_mean - expert_out_mean  + grad_penalty
        #self.total_loss  = generator_out_mean - expert_out_mean  
        self.losses = [generator_out_mean, expert_out_mean, grad_penalty]
        self.loss_name = ["generator_output", "expert_output",  "grad_penalty_loss"]
        # NOTE I am not sure how to scale the reward Build Reward for policy
        self.reward_op = 0.01 * generator_logits  

        # trainable var list
        var_list = self.get_trainable_variables()
        # self.lossandgrad = tf_util.function(
        #     [self.generator_obs_ph, self.generator_next_obs_ph, self.expert_obs_ph, self.expert_next_obs_ph],
        #     self.losses + [tf_util.flatgrad(self.total_loss, var_list)])

        gail_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.d_learning_rate)
        self.train_op=gail_optimizer.minimize(self.total_loss, var_list=var_list)
        # Log discriminator scalars for debugging purposes
        wgan_scalar_summaries = []
        for i, loss_name in enumerate(self.loss_name):
            i = tf.summary.scalar(loss_name, self.losses[i])
            wgan_scalar_summaries.append(i)
        self.wgan_summary = tf.summary.merge(wgan_scalar_summaries)


    def flatten(self, obs_ph, next_obs_ph, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            if self.normalize:
                with tf.variable_scope("obfilter"):
                    self.obs_rms = RunningMeanStd(shape=self.observation_shape)
                ## normalize obs and next_obs. NOTE do we need two smoothers ? --Judy
                obs = (obs_ph - self.obs_rms.mean) / self.obs_rms.std
                next_obs = (next_obs_ph - self.obs_rms.mean) / self.obs_rms.std
            else:
                obs = obs_ph
                next_obs = next_obs_ph


            ob_fl = tf.contrib.layers.flatten(obs)
            next_obs_fl = tf.contrib.layers.flatten(next_obs)
            flatten_input = tf.concat([ob_fl, next_obs_fl], axis=1)  # concatenate the two input -> form a transition
            return flatten_input

    def build_WS_graph(self, inputs, reuse=True):
        """
        build the graph, where observations are all flattened,
        in order to address image observations.

        :param reuse: (bool)
        :return: (tf.Tensor) the graph output
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


    def get_reward(self, obs, next_obs, sess=None):
        """
        Predict the reward using the observation and action

        :param obs: (tf.Tensor or np.ndarray) the observation
        :param actions: (tf.Tensor or np.ndarray) the action
        :return: (np.ndarray) the reward
        """
        # if sess is None:
        #     sess = tf.get_default_session()
        # if len(obs.shape) == 1:
        #     obs = np.expand_dims(obs, 0)
        # if len(actions.shape) == 1:
        #     actions = np.expand_dims(actions, 0)
        # elif len(actions.shape) == 0:
        #     # one discrete action
        #     actions = np.expand_dims(actions, 0)

        feed_dict = {self.generator_obs_ph: obs, self.generator_next_obs_ph: next_obs}
        reward = sess.run([self.reward_op], feed_dict)
        return reward


    def generate_discriminator_data(self, teacher_buffer, replay_buffer, batch_size, sess):
        expert_batch = teacher_buffer.sample(batch_size)
        # each sample consists of s, a, r, s', if_done
        ob_expert, next_ob_expert = expert_batch[0], expert_batch[3]
        batch_size = ob_expert.shape[0]
        batch = replay_buffer.sample(batch_size)
        ob_batch, next_ob_batch =  batch[0], batch[3]
        

        # update running mean/std for discriminator
        if self.normalize:
            self.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0),sess=sess)

        # Reshape actions if needed when using discrete actions

        return ob_expert, next_ob_expert, ob_batch, next_ob_batch


    def train_discriminator(self, writer, logger, step, d_gradient_steps, teacher_buffer, replay_buffer, batch_size, sess):

        logger.log("Optimizing Discriminator...")
        if step % 1000 == 0:
            logger.log(fmt_row(13, self.loss_name + ['discriminator-total-loss'] ))
        #timesteps_per_batch = len(observation)

        # NOTE: uses only the last g step for observation
        d_losses = []  # list of tuples, each of which gives the loss for a minibatch
        for i in range(d_gradient_steps):
            ob_expert, next_ob_expert, ob_batch, next_ob_batch = self.generate_discriminator_data(teacher_buffer, replay_buffer, batch_size, sess)
            feed_dict = {
                self.generator_obs_ph: ob_batch,
                self.generator_next_obs_ph: next_ob_batch,
                self.expert_obs_ph: ob_expert,
                self.expert_next_obs_ph: next_ob_expert
            }

            if writer is not None:
                run_ops = [self.wgan_summary] + self.losses + [self.total_loss, self.train_op]
                out = sess.run(run_ops, feed_dict)
                summary = out.pop(0)
                writer.add_summary(summary, step) 
            else:
                run_ops = self.losses + [self.total_loss, self.train_op]
                out = sess.run(run_ops, feed_dict)
            
            losses = out[:-1]

            d_losses.append(losses)
            del ob_batch, next_ob_batch, ob_expert, next_ob_expert
        if step % 1000 == 0:
            logger.log(fmt_row(13, np.mean(d_losses, axis=0)))


