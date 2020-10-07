"""
Reference: https://github.com/openai/imitation
I follow the architecture from the official repository
"""
import gym
import tensorflow as tf
import numpy as np
import random
from mpi4py import MPI
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

def allmean(arr, nworkers):
    assert isinstance(arr, np.ndarray)
    out = np.empty_like(arr)
    MPI.COMM_WORLD.Allreduce(arr, out, op=MPI.SUM)
    out /= nworkers
    return out

class SingleGAIfODiscriminator(object):
    def __init__(self, observation_space, hidden_size=128,
                 entcoeff=0.001, gradcoeff=10,scope="adversary", learning_rate=3e-4, normalize=True):
        """
        Reward regression from observations and transitions

        :param observation_space: (gym.spaces) 
        :param hidden_size: ([int]) the hidden dimension for the MLP
        :param entcoeff: (float) the entropy loss weight
        :param scope: (str) tensorflow variable scope
        :param normalize: (bool) Whether to normalize the reward or not
        """
        self.allmean = allmean
        # TODO: support images properly (using a CNN)
        self.scope = scope
        self.observation_shape = observation_space.shape
        self.observation_space = observation_space

        self.hidden_size = hidden_size
        self.normalize = normalize
        self.obs_rms = None

        # Placeholders
        self.generator_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape,
                                               name="w1_observations_ph")
        self.expert_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape,
                                            name="w1_expert_observations_ph")
        #self.generator_next_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape,
        #                                       name="w1_next_observations_ph")
        #self.expert_next_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape,
        #                                    name="w1_expert_next_observations_ph")
        self.d_learning_rate_ph = tf.placeholder(tf.float32, [], name="d_learning_rate_ph")
        # Build graph
        generator_input = self.flatten(self.generator_obs_ph, reuse=False)
        generator_input = tf.stop_gradient(generator_input)
        generator_logits = self.build_GAIfO_graph(generator_input, reuse=False)

        expert_input = self.flatten(self.expert_obs_ph, reuse=True)
        expert_input = tf.stop_gradient(expert_input)
        expert_logits = self.build_GAIfO_graph(expert_input, reuse=True) 

        # Build accuracy
        generator_acc = tf.reduce_mean(tf.cast(tf.nn.sigmoid(generator_logits) < 0.5, tf.float32))
        expert_acc = tf.reduce_mean(tf.cast(tf.nn.sigmoid(expert_logits) > 0.5, tf.float32))
        # Build regression loss
        sample_generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits, labels=tf.zeros_like(generator_logits))
        generator_loss = tf.reduce_mean(sample_generator_loss)
        sample_expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits, labels=tf.ones_like(expert_logits))
        expert_loss = tf.reduce_mean(sample_expert_loss)
        # Build entropy loss
        logits = tf.concat([generator_logits, expert_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        entropy_loss = -entcoeff * entropy
        # Build Gradient-penalty loss
        alpha_shape = self.observation_shape[0] 
        alpha = np.random.uniform(size=(1, alpha_shape))
        #inter = tf.multiply(generator_input ) + tf.multiply(expert_input, 1 - alpha) 
        inter = alpha * generator_input + (1 - alpha) * expert_input 
        with tf.GradientTape() as tape2:
            tape2.watch(inter)
            inter_output = self.build_GAIfO_graph(inter, reuse=True)
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
        p = tf.clip_by_value(1 - tf.nn.sigmoid(generator_logits), 1e-8, 1 - 1e-8)
        self.reward_op = -tf.log(p)
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
        # used for MPI training
        self.lossandgrad = tf_util.function(
            [self.generator_obs_ph, self.expert_obs_ph],
            [self.gail_summary] + self.losses + [tf_util.flatgrad(self.total_loss, var_list)])


    def flatten(self, obs_ph, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            if self.normalize:
                with tf.variable_scope("obfilter"):
                    self.obs_rms = RunningMeanStd(shape=self.observation_shape)
                ## normalize obs and next_obs. NOTE do we need two smoothers ? --Judy
                obs = (obs_ph - self.obs_rms.mean) / self.obs_rms.std
            else:
                obs = obs_ph


            ob_fl = tf.contrib.layers.flatten(obs)
            return ob_fl


    def build_GAIfO_graph(self, inputs, reuse=True):
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


    def get_reward(self, obs, sess=None):
        """
        Predict the reward using the observation and action

        :param obs: (tf.Tensor or np.ndarray) the observation
        :param actions: (tf.Tensor or np.ndarray) the action
        :return: (np.ndarray) the reward
        """
        feed_dict = {self.generator_obs_ph: obs}
        reward = sess.run([self.reward_op], feed_dict)
        return reward

    def generate_discriminator_data(self, teacher_buffer, replay_buffer, batch_size, sess):
        ob_expert = teacher_buffer.sample(batch_size)[0]
        # each sample consists of s, a, r, s', if_done
        ob_batch = replay_buffer.sample(batch_size)[0]
        print(ob_expert[0].shape, ob_batch[0].shape)
        
        # update running mean/std for discriminator
        if self.normalize:
            self.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0),sess=sess)
            #self.obs_rms.update(ob_expert, sess=sess)

        return ob_expert, ob_batch 

    def generate_onpolicy_discriminator_data(self, teacher_buffer, segs, batch_size, sess):
        ob_expert = teacher_buffer.sample(batch_size)[0]
        # each sample consists of s, a, r, s', if_done
        observations = segs["observations"] 
        indx = [random.randint(0, len(observations)-1) for _ in range(batch_size)]
        #next_indx = [i + 1 for i in indx]
        ob_batch = observations[indx]  
        
        # update running mean/std for discriminator
        if self.normalize:
            self.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0),sess=sess)
            #self.obs_rms.update(ob_expert, sess=sess)

        return ob_expert, ob_batch 




    
    def train_discriminator(self, writer, logger, step, d_gradient_steps, d_learning_rate, teacher_buffer, replay_buffer, batch_size, sess):

        logger.log("Optimizing Discriminator...")
        if step % 1000 == 0:
            logger.log(fmt_row(13, self.loss_name + ['discriminator-total-loss'] ))
        #timesteps_per_batch = len(observation)

        # NOTE: uses only the last g step for observation
        d_losses = []  # list of tuples, each of which gives the loss for a minibatch
        for i in range(d_gradient_steps):
            ob_expert, ob_batch = self.generate_discriminator_data(teacher_buffer, replay_buffer, batch_size, sess)
            feed_dict = {
                self.generator_obs_ph: ob_batch,
                self.expert_obs_ph: ob_expert,
                self.d_learning_rate_ph: d_learning_rate
            }

            if writer is not None:
                run_ops = [self.gail_summary] + self.losses + [self.total_loss, self.train_op]
                out = sess.run(run_ops, feed_dict)
                summary = out.pop(0)
                writer.add_summary(summary, step) 
            else:
                run_ops = self.losses + [self.total_loss, self.train_op]
                out = sess.run(run_ops, feed_dict)
            losses = out[:-1]
            d_losses.append(losses)

            del ob_batch, ob_expert

        if step % 1000 == 0:
            logger.log(fmt_row(13, np.mean(d_losses, axis=0)))


    def train_onpolicy_discriminator(self, writer, logger, d_gradient_steps, d_learning_rate, batch_size, teacher_buffer, replay_buffer, segs, num_timesteps, sess, d_adam, nworkers):

        logger.log("Optimizing Discriminator...")
        #if step % 1000 == 0:
        logger.log(fmt_row(13, self.loss_name ))
        #timesteps_per_batch = len(observation)

        # NOTE: uses only the last g step for observation
        d_losses = []  # list of tuples, each of which gives the loss for a minibatch
        #observations = seg["observations"] 
        for i in range(d_gradient_steps):
            steps = num_timesteps + (i + 1) * (segs["total_timestep"] / d_gradient_steps)
            if replay_buffer is None:
                ob_expert, ob_batch = self.generate_onpolicy_discriminator_data(teacher_buffer, segs, batch_size, sess)
            else:
                ob_expert, ob_batch = self.generate_discriminator_data(teacher_buffer, replay_buffer, batch_size, sess)
            gail_summary, *newlosses, grad = self.lossandgrad(ob_batch, ob_expert)
            d_adam.update(self.allmean(grad, nworkers), d_learning_rate)
            d_losses.append(newlosses)
            if writer is not None:
                writer.add_summary(gail_summary, steps)
            del ob_batch, ob_expert
        #if step % 1000 == 0:
        logger.log(fmt_row(13, np.mean(d_losses, axis=0)))


