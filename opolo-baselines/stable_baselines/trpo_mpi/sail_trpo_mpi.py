import time
from contextlib import contextmanager
from collections import deque

import gym
from mpi4py import MPI
import tensorflow as tf
import numpy as np

import stable_baselines.common.tf_util as tf_util
from stable_baselines.common import explained_variance, zipsame, dataset, fmt_row, colorize, ActorCriticRLModel, \
    SetVerbosity, TensorboardWriter
from stable_baselines import logger
from stable_baselines.common.mpi_adam import MpiAdam
from stable_baselines.common.cg import conjugate_gradient
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.gail.dataset.dataset import ExpertDataset
from stable_baselines.a2c.utils import total_episode_reward_logger, total_episode_logit_reward_logger 
from stable_baselines.trpo_mpi.utils import SegmentGenerator, add_vtarg_and_adv, flatten_lists
from stable_baselines.gail.adversary import DiscriminatorCalssifier
from stable_baselines.deepq.replay_buffer import ReplayBufferExtend 

class TRPO(ActorCriticRLModel):
    """
    Trust Region Policy Optimization (https://arxiv.org/abs/1502.05477)

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) the discount value
    :param timesteps_per_batch: (int) the number of timesteps to run per batch (horizon)
    :param max_kl: (float) the Kullback-Leibler loss threshold
    :param cg_iters: (int) the number of iterations for the conjugate gradient calculation
    :param lam: (float) GAE factor
    :param entcoeff: (float) the weight for the entropy loss
    :param cg_damping: (float) the compute gradient dampening factor
    :param vf_stepsize: (float) the value function stepsize
    :param vf_iters: (int) the value function's number iterations for learning
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    """

    def __init__(self, policy, env, gamma=0.99, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, lam=0.98,
                 entcoeff=0.0, cg_damping=1e-2, vf_stepsize=3e-4, vf_iters=3, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False,using_gail=False,expert_data_path=None):
        super(TRPO, self).__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=False,
                                   _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs)

        self.using_gail = using_gail
        self.timesteps_per_batch = timesteps_per_batch
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.gamma = gamma
        self.lam = lam
        self.max_kl = max_kl
        self.vf_iters = vf_iters
        self.vf_stepsize = vf_stepsize
        self.entcoeff = entcoeff
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log

        # GAIL & POfD Params
        self.hidden_size_adversary = 100
        self.adversary_entcoeff = 1e-3
        self.gradient_penalty_entcoeff = 10     
        self.expert_dataset = None
        self.g_step = 1
        self.d_step = 1
        self.d_stepsize = 3e-4
        self.shaping_mode=None

        # Lagrange Parmas
        self.demo_buffer_size = int(1e4)


        self.graph = None
        self.sess = None
        self.policy_pi = None
        self.loss_names = None
        self.assign_old_eq_new = None
        self.compute_losses = None
        self.compute_lossandgrad = None
        self.compute_fvp = None
        self.compute_vflossandgrad = None
        self.d_adam = None
        self.vfadam = None
        self.get_flat = None
        self.set_from_flat = None
        self.timed = None
        self.allmean = None
        self.nworkers = None
        self.rank = None
        self.discriminator = None
        self.step = None
        self.proba_step = None
        self.initial_state = None
        self.params = None
        self.summary = None
        self.episode_reward = None
        #self.using_gail = using_gail
        self.sparse_reward = False
        self.expert_data_path = expert_data_path
        if _init_setup_model:
            self.setup_model()
        
        #if using_gail:
        if self.using_gail:
            self.expert_dataset = ExpertDataset(expert_path=self.expert_data_path, ob_flatten=False)
            print('-'*20 + "expert_data_path: {}".format(self.expert_data_path))
            time.sleep(4)
            self.expert_scores = []

    def _get_pretrain_placeholders(self):
        policy = self.policy_pi
        action_ph = policy.pdtype.sample_placeholder([None])
        if isinstance(self.action_space, gym.spaces.Discrete):
            return policy.obs_ph, action_ph, policy.policy
        return policy.obs_ph, action_ph, policy.deterministic_action

    def setup_model(self):
        # prevent import loops
        

        with SetVerbosity(self.verbose):

            assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the TRPO model must be " \
                                                               "an instance of common.policies.ActorCriticPolicy."

            self.nworkers = MPI.COMM_WORLD.Get_size()
            self.rank = MPI.COMM_WORLD.Get_rank()
            np.set_printoptions(precision=3)

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_util.single_threaded_session(graph=self.graph)

                if self.using_gail:
                    self.discriminator = DiscriminatorCalssifier(
                        self.observation_space,
                        self.action_space,
                        self.hidden_size_adversary,
                        entcoeff=self.adversary_entcoeff,
                        gradcoeff=self.gradient_penalty_entcoeff)

                # Construct network for new policy
                self.policy_pi = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                             None, reuse=False, **self.policy_kwargs)

                # Network for old policy
                with tf.variable_scope("oldpi", reuse=False):
                    old_policy = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                             None, reuse=False, **self.policy_kwargs)

                with tf.variable_scope("loss", reuse=False):
                    atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
                    ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

                    observation = self.policy_pi.obs_ph
                    action = self.policy_pi.pdtype.sample_placeholder([None])

                    kloldnew = old_policy.proba_distribution.kl(self.policy_pi.proba_distribution)
                    ent = self.policy_pi.proba_distribution.entropy()
                    meankl = tf.reduce_mean(kloldnew)
                    meanent = tf.reduce_mean(ent)
                    entbonus = self.entcoeff * meanent

                    vferr = tf.reduce_mean(tf.square(self.policy_pi.value_flat - ret))
                    # advantage * pnew / pold
                    ratio = tf.exp(self.policy_pi.proba_distribution.logp(action) -
                                   old_policy.proba_distribution.logp(action))
                    surrgain = tf.reduce_mean(ratio * atarg)

                    optimgain = surrgain + entbonus
                    losses = [optimgain, meankl, entbonus, surrgain, meanent]
                    self.loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

                    dist = meankl

                    all_var_list = tf_util.get_trainable_vars("model")
                    var_list = [v for v in all_var_list if "/vf" not in v.name and "/q/" not in v.name]
                    vf_var_list = [v for v in all_var_list if "/pi" not in v.name and "/logstd" not in v.name]

                    self.get_flat = tf_util.GetFlat(var_list, sess=self.sess)
                    self.set_from_flat = tf_util.SetFromFlat(var_list, sess=self.sess)

                    klgrads = tf.gradients(dist, var_list)
                    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
                    shapes = [var.get_shape().as_list() for var in var_list]
                    start = 0
                    tangents = []
                    for shape in shapes:
                        var_size = tf_util.intprod(shape)
                        tangents.append(tf.reshape(flat_tangent[start: start + var_size], shape))
                        start += var_size
                    gvp = tf.add_n([tf.reduce_sum(grad * tangent)
                                    for (grad, tangent) in zipsame(klgrads, tangents)])  # pylint: disable=E1111
                    fvp = tf_util.flatgrad(gvp, var_list)

                    tf.summary.scalar('entropy_loss', meanent)
                    tf.summary.scalar('policy_gradient_loss', optimgain)
                    tf.summary.scalar('value_function_loss', surrgain)
                    tf.summary.scalar('approximate_kullback-leibler', meankl)
                    tf.summary.scalar('loss', optimgain + meankl + entbonus + surrgain + meanent)

                    self.assign_old_eq_new = \
                        tf_util.function([], [], updates=[tf.assign(oldv, newv) for (oldv, newv) in
                                                          zipsame(tf_util.get_globals_vars("oldpi"),
                                                                  tf_util.get_globals_vars("model"))])
                    self.compute_losses = tf_util.function([observation, old_policy.obs_ph, action, atarg], losses)
                    self.compute_fvp = tf_util.function([flat_tangent, observation, old_policy.obs_ph, action, atarg],
                                                        fvp)
                    self.compute_vflossandgrad = tf_util.function([observation, old_policy.obs_ph, ret],
                                                                  tf_util.flatgrad(vferr, vf_var_list))

                    @contextmanager
                    def timed(msg):
                        if self.rank == 0 and self.verbose >= 1:
                            print(colorize(msg, color='magenta'))
                            start_time = time.time()
                            yield
                            print(colorize("done in {:.3f} seconds".format((time.time() - start_time)),
                                           color='magenta'))
                        else:
                            yield

                    def allmean(arr):
                        assert isinstance(arr, np.ndarray)
                        out = np.empty_like(arr)
                        MPI.COMM_WORLD.Allreduce(arr, out, op=MPI.SUM)
                        out /= self.nworkers
                        return out

                    tf_util.initialize(sess=self.sess)

                    th_init = self.get_flat()
                    MPI.COMM_WORLD.Bcast(th_init, root=0)
                    self.set_from_flat(th_init)

                with tf.variable_scope("Adam_mpi", reuse=False):
                    self.vfadam = MpiAdam(vf_var_list, sess=self.sess)
                    if self.using_gail:
                        self.d_adam = MpiAdam(self.discriminator.get_trainable_variables(), sess=self.sess)
                        self.d_adam.sync()
                    self.vfadam.sync()

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('discounted_rewards', tf.reduce_mean(ret))
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.vf_stepsize))
                    tf.summary.scalar('advantage', tf.reduce_mean(atarg))
                    tf.summary.scalar('kl_clip_range', tf.reduce_mean(self.max_kl))

                    if self.full_tensorboard_log:
                        tf.summary.histogram('discounted_rewards', ret)
                        tf.summary.histogram('learning_rate', self.vf_stepsize)
                        tf.summary.histogram('advantage', atarg)
                        tf.summary.histogram('kl_clip_range', self.max_kl)
                        if tf_util.is_image(self.observation_space):
                            tf.summary.image('observation', observation)
                        else:
                            tf.summary.histogram('observation', observation)

                self.timed = timed
                self.allmean = allmean

                self.step = self.policy_pi.step
                self.proba_step = self.policy_pi.proba_step
                self.initial_state = self.policy_pi.initial_state

                self.params = tf_util.get_trainable_vars("model") + tf_util.get_trainable_vars("oldpi")
                if self.using_gail:
                    self.params.extend(self.discriminator.get_trainable_variables())

                self.summary = tf.summary.merge_all()

                self.compute_lossandgrad = \
                    tf_util.function([observation, old_policy.obs_ph, action, atarg, ret],
                                     [self.summary, tf_util.flatgrad(optimgain, var_list)] + losses)


    def learn(self, total_timesteps, callback=None, seed=None, test_env=None, log_interval=100, tb_log_name="TRPO",
              reset_num_timesteps=True, config=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        ############################
        # change entcoeff is necessary
        ############################
        if config is not None:
            self.config = config
            self.entcoeff = config.get('entropy_lamda', self.entcoeff)
            self.shaping_mode = config.get('shaping_mode', None)
            print('Entropy coeff: {}'.format(self.entcoeff))
            print('Shaping model: {}'.format(self.shaping_mode))
            time.sleep(3)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn(seed)

            with self.sess.as_default():
                seg_generator = SegmentGenerator(
                    self.policy_pi,
                    self.env,
                    self.timesteps_per_batch,
                    discriminator=self.discriminator,
                    sess=self.sess,
                    config=config
                )
                seg_gen = seg_generator.traj_segment_generator_pro() 
                episodes_so_far = 0
                timesteps_so_far = 0
                iters_so_far = 0
                t_start = time.time()
                len_buffer = deque(maxlen=40)  # rolling buffer for episode lengths
                reward_buffer = deque(maxlen=40)  # rolling buffer for episode rewards
                self.episode_reward = np.zeros((self.n_envs,))
                self.episode_logit = np.zeros((self.n_envs,))

                true_reward_buffer = deque(maxlen=40)
                sparse_reward_buffer = deque(maxlen=40)
                logit_buffer = deque(maxlen=40)
                if self.using_gail:

                    # Initialize dataloader
                    self.demo_replay_buffer = ReplayBufferExtend(self.demo_buffer_size)
                    self.initialize_expert_buffer()

                while True:
                    if callback is not None:
                        # Only stop training if return value is False, not when it is None. This is for backwards
                        # compatibility with callbacks that have no return statement.
                        if callback(locals(), globals()) is False:
                            break
                    if total_timesteps and timesteps_so_far >= total_timesteps:
                        break

                    logger.log("********** Iteration %i ************" % iters_so_far)

                    def fisher_vector_product(vec):
                        return self.allmean(self.compute_fvp(vec, *fvpargs, sess=self.sess)) + self.cg_damping * vec

                    # ------------------ Update G ------------------
                    logger.log("Optimizing Policy...")
                    # g_step = 1 when not using GAIL
                    mean_losses = None
                    vpredbefore = None
                    tdlamret = None
                    observation = None
                    action = None
                    seg = None
                    for k in range(self.g_step):
                        with self.timed("sampling"):
                            seg = seg_gen.__next__()
                        add_vtarg_and_adv(seg, self.gamma, self.lam, self.shaping_mode)
                        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
                        observation, action = seg["observations"], seg["actions"]
                        atarg, tdlamret = seg["adv"], seg["tdlamret"]


                        vpredbefore = seg["vpred"]  # predicted value function before update
                        atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate

                        # true_rew is the reward without discount
                        if writer is not None:
                            self.episode_reward, self.episode_logit = total_episode_logit_reward_logger(
                                self.episode_reward,
                                self.episode_logit,
                                seg["true_rewards"].reshape((self.n_envs, -1)),
                                #seg["ep_logit_rets"].reshape((self.n_envs, -1)),
                                np.array(seg["ep_logit_rets"]).reshape((self.n_envs, -1)),
                                seg["dones"].reshape((self.n_envs, -1)),
                                writer,
                                self.num_timesteps)

                        args = seg["observations"], seg["observations"], seg["actions"], atarg
                        # Subsampling: see p40-42 of John Schulman thesis
                        # http://joschu.net/docs/thesis.pdf
                        fvpargs = [arr[::5] for arr in args]

                        self.assign_old_eq_new(sess=self.sess)

                        with self.timed("computegrad"):
                            steps = self.num_timesteps + (k + 1) * (seg["total_timestep"] / self.g_step)
                            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                            run_metadata = tf.RunMetadata() if self.full_tensorboard_log else None
                            # run loss backprop with summary, and save the metadata (memory, compute time, ...)
                            if writer is not None:
                                summary, grad, *lossbefore = self.compute_lossandgrad(*args, tdlamret, sess=self.sess,
                                                                                      options=run_options,
                                                                                      run_metadata=run_metadata)
                                if self.full_tensorboard_log:
                                    writer.add_run_metadata(run_metadata, 'step%d' % steps)
                                writer.add_summary(summary, steps)
                            else:
                                _, grad, *lossbefore = self.compute_lossandgrad(*args, tdlamret, sess=self.sess,
                                                                                options=run_options,
                                                                                run_metadata=run_metadata)

                        lossbefore = self.allmean(np.array(lossbefore))
                        grad = self.allmean(grad)
                        if np.allclose(grad, 0):
                            logger.log("Got zero gradient. not updating")
                        else:
                            with self.timed("conjugate_gradient"):
                                stepdir = conjugate_gradient(fisher_vector_product, grad, cg_iters=self.cg_iters,
                                                             verbose=self.rank == 0 and self.verbose >= 1)
                            assert np.isfinite(stepdir).all()
                            shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
                            # abs(shs) to avoid taking square root of negative values
                            lagrange_multiplier = np.sqrt(abs(shs) / self.max_kl)
                            # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
                            fullstep = stepdir / lagrange_multiplier
                            expectedimprove = grad.dot(fullstep)
                            surrbefore = lossbefore[0]
                            stepsize = 1.0
                            thbefore = self.get_flat()
                            thnew = None
                            for _ in range(10):
                                thnew = thbefore + fullstep * stepsize
                                self.set_from_flat(thnew)
                                mean_losses = surr, kl_loss, *_ = self.allmean(
                                    np.array(self.compute_losses(*args, sess=self.sess)))
                                improve = surr - surrbefore
                                logger.log("Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
                                if not np.isfinite(mean_losses).all():
                                    logger.log("Got non-finite value of losses -- bad!")
                                elif kl_loss > self.max_kl * 1.5:
                                    logger.log("violated KL constraint. shrinking step.")
                                elif improve < 0:
                                    logger.log("surrogate didn't improve. shrinking step.")
                                else:
                                    logger.log("Stepsize OK!")
                                    break
                                stepsize *= .5
                            else:
                                logger.log("couldn't compute a good step")
                                self.set_from_flat(thbefore)
                            if self.nworkers > 1 and iters_so_far % 20 == 0:
                                # list of tuples
                                paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), self.vfadam.getflat().sum()))
                                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

                        with self.timed("vf"):
                            for _ in range(self.vf_iters):
                                # NOTE: for recurrent policies, use shuffle=False?
                                for (mbob, mbret) in dataset.iterbatches((seg["observations"], seg["tdlamret"]),
                                                                         include_final_partial_batch=False,
                                                                         batch_size=128,
                                                                         shuffle=True):
                                    grad = self.allmean(self.compute_vflossandgrad(mbob, mbob, mbret, sess=self.sess))
                                    self.vfadam.update(grad, self.vf_stepsize)

                    if mean_losses is not None:
                        for (loss_name, loss_val) in zip(self.loss_names, mean_losses):
                            logger.record_tabular(loss_name, loss_val)

                    logger.record_tabular("explained_variance_tdlam_before",
                                          explained_variance(vpredbefore, tdlamret))

                    if self.using_gail:
                        self.train_discriminator(logger, observation, action)
                    
                    # ------------------ evaluating -------------------------------------
                    if test_env is not None:
                        self.eval(tet_env, deterministic=True, episodes=1, render=False, callback=None)

                    # ------------------  logging and Update Constraint------------------ 
                    # lr: lengths and rewards
                    lr_local = (seg["ep_lens"], seg["ep_rets"], seg["ep_true_rets"], seg["ep_sparse_rets"], seg["ep_logit_rets"])  # local values
                    list_lr_pairs = MPI.COMM_WORLD.allgather(lr_local)  # list of tuples
                    lens, rews, true_rets, sparse_rets, logit_rets = map(flatten_lists, zip(*list_lr_pairs))
                    true_reward_buffer.extend(true_rets)
                    sparse_reward_buffer.extend(sparse_rets)
                    logit_buffer.extend(logit_rets)
                    len_buffer.extend(lens)
                    reward_buffer.extend(rews)

                    if len(len_buffer) > 0:
                        logger.record_tabular("EpLenMean", np.mean(len_buffer))
                        logger.record_tabular("EpRewMean", np.mean(reward_buffer))
                        logger.record_tabular("EpTrueRewMean", np.mean(true_reward_buffer))
                        logger.record_tabular("EpSparseRewMean", np.mean(sparse_reward_buffer))
                        logger.record_tabular("EpLogitMean", np.mean(logit_buffer))
                    logger.record_tabular("EpThisIter", len(lens))
                    episodes_so_far += len(lens)
                    current_it_timesteps = MPI.COMM_WORLD.allreduce(seg["total_timestep"])
                    timesteps_so_far += current_it_timesteps
                    self.num_timesteps += current_it_timesteps
                    iters_so_far += 1

                    ### updating lagrange lambda 
                    reward = np.mean(true_reward_buffer)
                    gradient_lambda = self.cost_func(reward, config) # the larger the cost, the smaller the real reward
                    #self.lagrange_lambda = self.lagrange_lambda + self.lagrange_stepsize * gradient_lambda
                    lagrange_lambda = seg_generator.set_lagrange_lambda(gradient_lambda)


                    logger.record_tabular("EpisodesSoFar", episodes_so_far)
                    logger.record_tabular("TimestepsSoFar", self.num_timesteps)
                    logger.record_tabular("TimeElapsed", time.time() - t_start)
                    logger.record_tabular("Lagrange Lambda", lagrange_lambda)

                    if self.verbose >= 1 and self.rank == 0:
                        logger.dump_tabular()

        return self

    def eval(self, env, deterministic=True, episodes=1, render=False, callback=None):
        episode_rewards, n_steps = [], 0
        for _ in range(episodes):
            obs = env.reset()
            done, state = False, None
            episode_reward = 0.0
            while not done:
                # get action prediction from model
                action = self.policy_tf.step(obs).flatten()
                # Add noise to the action if not deterministic
                if not deterministic and self.action_noise is not None:
                    noise = self.action_noise()
                    action = np.clip(action + noise, -1, 1)
                # Rescale from [-1, 1] to the correct bounds
                action = action * np.abs(self.action_space.low)
                assert action.shape == env.action_space.shape

                # sample from environment with that action
                obs, reward, done, _info = env.step(action)
                episode_reward += reward
                if callback is not None:
                    callback(locals(), globals())
                n_steps += 1
                if render:
                    env.render()
            episode_rewards.append(episode_reward)
        mean_reward = np.mean(episode_rewards)
        return mean_reward, n_steps

    def save(self, save_path, cloudpickle=False):
        if self.using_gail and self.expert_dataset is not None:
            # Exit processes to pickle the dataset
            self.expert_dataset.prepare_pickling()
        data = {
            "gamma": self.gamma,
            "timesteps_per_batch": self.timesteps_per_batch,
            "max_kl": self.max_kl,
            "cg_iters": self.cg_iters,
            "lam": self.lam,
            "entcoeff": self.entcoeff,
            "cg_damping": self.cg_damping,
            "vf_stepsize": self.vf_stepsize,
            "vf_iters": self.vf_iters,
            "hidden_size_adversary": self.hidden_size_adversary,
            "adversary_entcoeff": self.adversary_entcoeff,
            "expert_dataset": self.expert_dataset,
            "g_step": self.g_step,
            "d_step": self.d_step,
            "d_stepsize": self.d_stepsize,
            "using_gail": self.using_gail,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)

    def cost_func(self, x, config):
        """
        Sigmoid scale function.
        Input: x: the unnormalized reward. 
        c: the maximal reward
        Return : alpha = 1 - sigmoid(x) , 0 < alpha < 1
        """ 
        c = config['max_score']
        #norm_reward = np.clip(x / c, 0, 1)
        cost = (c - x) /c
        return 1.0 - 1.0 / (1 + np.exp(cost)) 

    def initialize_expert_buffer(self):
        demo_obs, demo_actions, demo_rewards, demo_dones, demo_next_obs, demo_episode_scores = self.expert_dataset.get_transitions()   
        episode_lengths = np.where(demo_dones == 1)[0]
        n_samples = len(demo_obs)
        # get episode_score for each demo sample, either 0 or episode-reward
        episode_idx, n_episodes = 0, len(demo_episode_scores)
        for idx in range(n_samples-1):
            episode_score = demo_episode_scores[episode_idx] 
            episode_length = episode_lengths[episode_idx] 
            es = self.normalize_score(episode_score)
            discount_return = es * (self.gamma ** (episode_length - idx)) 
            ### replace reward to be sparse if necessary
            true_reward = demo_rewards[idx]
            if self.sparse_reward:
                true_reward = 0 if demo_dones[idx] == 0 else episode_score 
                if self.norm_sparse_reward:
                    true_reward = 0 if demo_dones[idx] == 0 else es  

            if demo_dones[idx+1] == 1:
                print(idx, 'episode_score for demonstration tarjectory: {}'.format(episode_score)) 
                episode_idx += 1 
                self.expert_scores.append(episode_score)
                assert episode_length - idx >= 0
            
            self.demo_replay_buffer.add(demo_obs[idx], demo_actions[idx], demo_rewards[idx], demo_obs[idx + 1], float(demo_dones[idx]), 1.0, discount_return, true_reward)
            if idx % 1000 == 0:
                print("Adding demonstration to the replay buffer, processing {} %  ..".format(float(idx+1) * 100 / n_samples))
        true_reward = demo_rewards[-1]
        ### add last sample to buffer
        if self.sparse_reward:
            true_reward = episode_score
            if self.norm_sparse_reward:
                true_reward = self.normalize_score(episode_score)
        self.demo_replay_buffer.add(demo_obs[-1], demo_actions[-1], demo_rewards[-1], demo_obs[-1], float(demo_dones[-1]), 1.0, es, true_reward)

    def normalize_score(self, episode_score):
        min_score = self.config['optimal_score']
        assert min_score != 0
        return 1 + (episode_score - min_score) / np.abs(min_score)
    
    def train_discriminator(self, logger, observation, action):
        # ------------------ Update D ------------------
        logger.log("Optimizing Discriminator...")
        logger.log(fmt_row(13, self.discriminator.loss_name))
        assert len(observation) == self.timesteps_per_batch
        batch_size = 100
        gradient_steps = self.timesteps_per_batch // batch_size

        # NOTE: uses only the last g step for observation
        d_losses = []  # list of tuples, each of which gives the loss for a minibatch
        # NOTE: for recurrent policies, use shuffle=False?
        for ob_batch, ac_batch in dataset.iterbatches((observation, action),
                                                      include_final_partial_batch=False,
                                                      batch_size=batch_size,
                                                      shuffle=True):
            ob_expert, ac_expert = self.demo_replay_buffer.sample(batch_size)[:2]
            ###### upsample expert if necessary 
            if self.discriminator.normalize:
                self.discriminator.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))

            # Reshape actions if needed when using discrete actions
            if isinstance(self.action_space, gym.spaces.Discrete):
                if len(ac_batch.shape) == 2:
                    ac_batch = ac_batch[:, 0]
                if len(ac_expert.shape) == 2:
                    ac_expert = ac_expert[:, 0]
            *newlosses, grad = self.discriminator.lossandgrad(ob_batch, ac_batch, ob_expert, ac_expert)
            self.d_adam.update(self.allmean(grad), self.d_stepsize)
            d_losses.append(newlosses)
        logger.log(fmt_row(13, np.mean(d_losses, axis=0)))