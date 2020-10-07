import random
import numpy as np
from stable_baselines.common.segment_tree import SumSegmentTree, MinSegmentTree
from collections import deque

class TeacherBuffer(object):
    """
    prioritized trajectories based on their episodic scores
    """
    def __init__(self, max_episodes=10):
        self._max_episodes = max_episodes
        self.episodes = 0
        self._storage = {i:None for i in range(max_episodes)}
        self._episode_steps = {i:0 for i in range(max_episodes)}
        self.episode_scores = []
        self.N = np.sum(list(self._episode_steps.values()))

    def update_trajectory(self, trajectory, score):
        """
        trajectory contains list of tuples
        """
        if self.episodes < self._max_episodes:
            # fill in buffer first if we haven't reach the threshold
            self._storage[self.episodes] = [i for i in trajectory]  # add trajectory
            self._episode_steps[self.episodes] = len(trajectory)
            self.episodes += 1
            self.episode_scores.append(score)
        else:
            # find trajectory with lowest score
            idx = np.argmin(self.episode_scores)
            threshold_score = np.min(self.episode_scores)
            if score < threshold_score:
                # do not update buffer if new trajectory is no better than previous trajectories
                return 
            # cleanup old trajectory
            del self._storage[idx]
            # reset with new trajectory
            self._storage[idx] = [i for i in trajectory]
            self._episode_steps[idx] = len(trajectory)
            self.episode_scores[idx] = score

        # update N
        self.N = np.sum(list(self._episode_steps.values()))
        return np.mean(self.episode_scores)

    def get_threshold(self):
        if self.episodes == 0:
            return None        
        return np.argmin(self.episode_scores)
    
    def sample(self, batch_size):
        proportions = [ int(batch_size * float(self._episode_steps[i]) / self.N ) for i in range(self.episodes)]
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], [] 

        # get trajectory index (j) and sample index in that trajectory (i)
        for j, p in enumerate(proportions): 
            idxes = [random.randint(0, self._episode_steps[j] - 1) for _ in range(p)]
            for i in idxes:
                data = self._storage[j][i]
                obs_t, action, reward, obs_tp1, done = data
                obses_t.append(np.array(obs_t, copy=False))
                actions.append(np.array(action, copy=False))
                rewards.append(reward)
                obses_tp1.append(np.array(obs_tp1, copy=False))
                dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones).reshape(-1,1) 

    def can_sample(self, batch_size):
        if self.episodes == 0:
            return False
        return self.N > batch_size


class TrajectoryBuffer(object):
    def __init__(self, max_size, gamma=0.99):
        self._storage = []
        self._next_idx = 0
        self._maxsize = int(max_size)
        self.gamma = gamma

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        :param episode_score: (float) is the episode score, only available when done == True
        """
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = int((self._next_idx + 1) % self._maxsize)
    
    def reset(self):
        """
        Free space, move pointers
        """
        del self._storage
        self._storage = []
        self._next_idx = 0
    
    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones).reshape(-1, 1)

    def sample(self, batch_size, **_kwargs):
        """
        Sample a batch of experiences.

        :param batch_size: (int) How many transitions to sample.
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def get_episode(self, **_kwargs):
        """
        get current episode
        """
        return [self._storage[i] for i in range(len(self._storage))]
    


class NStepReplayBuffer(object):
    def __init__(self, size, n_step, demo_size=0, gamma=0.99):
        """
        Implements a ring buffer (FIFO).

        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        """
        self._storage = [] 
        self._next_idx = 0
        self.demo_size = demo_size 
        self._maxsize = size + self.demo_size 
        self.queue = deque([], maxlen= n_step) # cache most recent k steps
        self.n_step = n_step
        self.gamma = gamma

    def __len__(self):
        return len(self._storage)

    @property
    def storage(self):
        """[(Union[np.ndarray, int], Union[np.ndarray, int], float, Union[np.ndarray, int], bool)]: content of the replay buffer"""
        return self._storage

    @property
    def buffer_size(self):
        """float: Max capacity of the buffer"""
        return self._maxsize

    def can_sample(self, n_samples):
        """
        Check if n_samples samples can be sampled
        from the buffer.

        :param n_samples: (int)
        :return: (bool)
        """
        return len(self) >= n_samples

    def is_full(self):
        """
        Check whether the replay buffer is full or not.

        :return: (bool)
        """
        return len(self) == self.buffer_size 


    def get_n_step_info(self):
        # (obs_t, action, reward, obs_tp1, done)
        # get last n-th step
        reward, next_state, done = self.queue[-1][-3:]
        # for the previous n-1 steps
        for transition in reversed(list(self.queue)[:-1]):
            r, n_s, d = transition[-3:]
            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else(next_state, done)
        return reward, next_state, done 

    def add_transition(self, obs_t, action, reward, obs_tp1, done, demo=False):
        # add data to n-step cache 
        cache = (obs_t, action, reward, obs_tp1, done)
        self.queue.append(cache)
        # cache is not ready
        if len(self.queue) < self.n_step:
            return None 
        # get the following (s_t + s_{t+1} + ... + s_{t+n-1}, s_{t+n}, done)
        r, ns, if_done = self.get_n_step_info()
        # get s_t
        current_state, current_action = self.queue[0][:2]
        self.add(current_state, current_action, r, ns, done, demo=demo)
        return self.queue[0] # return a single step transition

    def add(self, obs_t, action, reward, obs_tp1, done, demo=False):
        """
        add a new transition to the buffer

        :param obs_t: (Union[np.ndarray, int]) the last observation
        :param action: (Union[np.ndarray, int]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Union[np.ndarray, int]) the current observation
        :param done: (bool) is the episode done
        """
        data = (obs_t, action, reward, obs_tp1, done, float(demo))
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
        if demo and self._next_idx == 0: # only put demonstrations in the beginning of the replay buffer and never be replaced out
            self._next_idx = self.demo_size

    def extend(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new batch of transitions to the buffer

        :param obs_t: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the last batch of observations
        :param action: (Union[Tuple[Union[np.ndarray, int]]], np.ndarray]) the batch of actions
        :param reward: (Union[Tuple[float], np.ndarray]) the batch of the rewards of the transition
        :param obs_tp1: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the current batch of observations
        :param done: (Union[Tuple[bool], np.ndarray]) terminal status of the batch

        Note: uses the same names as .add to keep compatibility with named argument passing
                but expects iterables and arrays with more than 1 dimensions
        """
        for data in zip(obs_t, action, reward, obs_tp1, done):
            if self._next_idx >= len(self._storage):
                self._storage.append(data)
            else:
                self._storage[self._next_idx] = data
            self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones, if_demos = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, if_demo = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            if_demos.append(if_demo)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), np.array(if_demos)

    def sample(self, batch_size, **_kwargs):
        """
        Sample a batch of experiences.

        :param batch_size: (int) How many transitions to sample.
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
    
    def sample_by_index(self, idxes):
        return self._encode_sample(idxes)  


        

class ReplayBuffer(object):
    def __init__(self, size):
        """
        Implements a ring buffer (FIFO).

        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    @property
    def storage(self):
        """[(Union[np.ndarray, int], Union[np.ndarray, int], float, Union[np.ndarray, int], bool)]: content of the replay buffer"""
        return self._storage

    @property
    def buffer_size(self):
        """float: Max capacity of the buffer"""
        return self._maxsize

    def can_sample(self, n_samples):
        """
        Check if n_samples samples can be sampled
        from the buffer.

        :param n_samples: (int)
        :return: (bool)
        """
        return len(self) >= n_samples

    def is_full(self):
        """
        Check whether the replay buffer is full or not.

        :return: (bool)
        """
        return len(self) == self.buffer_size

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer

        :param obs_t: (Union[np.ndarray, int]) the last observation
        :param action: (Union[np.ndarray, int]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Union[np.ndarray, int]) the current observation
        :param done: (bool) is the episode done
        """
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = int((self._next_idx + 1) % self._maxsize)

    def extend(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new batch of transitions to the buffer

        :param obs_t: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the last batch of observations
        :param action: (Union[Tuple[Union[np.ndarray, int]]], np.ndarray]) the batch of actions
        :param reward: (Union[Tuple[float], np.ndarray]) the batch of the rewards of the transition
        :param obs_tp1: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the current batch of observations
        :param done: (Union[Tuple[bool], np.ndarray]) terminal status of the batch

        Note: uses the same names as .add to keep compatibility with named argument passing
                but expects iterables and arrays with more than 1 dimensions
        """
        for data in zip(obs_t, action, reward, obs_tp1, done):
            if self._next_idx >= len(self._storage):
                self._storage.append(data)
            else:
                self._storage[self._next_idx] = data
            self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        #return np.array(obses_t), np.array(actions), np.array(rewards).reshape(-1,1), np.array(obses_tp1), np.array(dones).reshape(-1,1)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size, **_kwargs):
        """
        Sample a batch of experiences.

        :param batch_size: (int) How many transitions to sample.
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        obs, acs, rewards, next_obs, dones = self._encode_sample(idxes)
        return obs, acs, rewards.reshape(-1,1), next_obs, dones.reshape(-1,1)
        
    def initialize_teacher_buffer(self, expert_dataset):
        demo_obs, demo_actions, demo_rewards, demo_dones, demo_next_obs, demo_episode_scores = expert_dataset.get_transitions()
        episode_lengths = np.where(demo_dones == 1)[0]
        n_samples = len(demo_obs)
        # get episode_score for each demo sample, either 0 or episode-reward
        episode_idx, n_episodes = 0, len(demo_episode_scores)
        for idx in range(n_samples-1):
            episode_score = demo_episode_scores[episode_idx]
            episode_length = episode_lengths[episode_idx]

            if demo_dones[idx+1] == 1:
                print('{}-th sample, episode_score for demonstration tarjectory: {}'.format(idx, episode_score))
                episode_idx += 1
                assert episode_length - idx >= 0
            #scaled_demo_action = scale_action(self.action_space, demo_actions[idx])
            self.add(demo_obs[idx], demo_actions[idx], demo_rewards[idx], demo_next_obs[idx], float(demo_dones[idx]))

            if idx % 1000 == 0:
                print("Adding demonstration to the replay buffer, processing {} %  ..".format(float(idx+1) * 100 / n_samples))
        ### add last sample to buffer
        #scaled_demo_action = scale_action(self.action_space, demo_actions[-1])
        self.add(demo_obs[-1], demo_actions[-1], demo_rewards[-1], demo_next_obs[-1], float(demo_dones[-1])) 


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """
        Create Prioritized Replay buffer.
        See Also ReplayBuffer.__init__
        :param size: (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories
            are dropped.
        :param alpha: (float) how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer
        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        idx = self._next_idx
        super().add(obs_t, action, reward, obs_tp1, done)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def extend(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new batch of transitions to the buffer
        :param obs_t: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the last batch of observations
        :param action: (Union[Tuple[Union[np.ndarray, int]]], np.ndarray]) the batch of actions
        :param reward: (Union[Tuple[float], np.ndarray]) the batch of the rewards of the transition
        :param obs_tp1: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the current batch of observations
        :param done: (Union[Tuple[bool], np.ndarray]) terminal status of the batch
        Note: uses the same names as .add to keep compatibility with named argument passing
            but expects iterables and arrays with more than 1 dimensions
        """
        idx = self._next_idx
        super().extend(obs_t, action, reward, obs_tp1, done)
        while idx != self._next_idx:
            self._it_sum[idx] = self._max_priority ** self._alpha
            self._it_min[idx] = self._max_priority ** self._alpha
            idx = (idx + 1) % self._maxsize

    def _sample_proportional(self, batch_size):
        mass = []
        total = self._it_sum.sum(0, len(self._storage) - 1)
        # TODO(szymon): should we ensure no repeats?
        mass = np.random.random(size=batch_size) * total
        idx = self._it_sum.find_prefixsum_idx(mass)
        return idx

    def sample(self, batch_size: int, beta: float = 0):
        """
        Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        :param batch_size: (int) How many transitions to sample.
        :param beta: (float) To what degree to use importance weights (0 - no corrections, 1 - full correction)
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
            - weights: (numpy float) Array of shape (batch_size,) and dtype np.float32 denoting importance weight of
                each sampled transition
            - idxes: (numpy int) Array of shape (batch_size,) and dtype np.int32 idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)
        p_sample = self._it_sum[idxes] / self._it_sum.sum()
        weights = (p_sample * len(self._storage)) ** (-beta) / max_weight
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """
        Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        assert np.min(priorities) > 0
        assert np.min(idxes) >= 0
        assert np.max(idxes) < len(self.storage)
        self._it_sum[idxes] = priorities ** self._alpha
        self._it_min[idxes] = priorities ** self._alpha

        self._max_priority = max(self._max_priority, np.max(priorities))

class NStepPrioritizedReplayBuffer(NStepReplayBuffer):
    def __init__(self, size, alpha, n_step=1, gamma=0.99, demo_size=0):
        """
        Create Prioritized Replay buffer.

        See Also ReplayBuffer.__init__

        :param size: (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories
            are dropped.
        :param alpha: (float) how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        super(NStepPrioritizedReplayBuffer, self).__init__(size, n_step, demo_size, gamma)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < self._maxsize:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0  
  
    def add(self, obs_t, action, reward, obs_tp1, done, demo=False):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        idx = self._next_idx
        super().add(obs_t, action, reward, obs_tp1, done, demo)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def extend(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new batch of transitions to the buffer

        :param obs_t: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the last batch of observations
        :param action: (Union[Tuple[Union[np.ndarray, int]]], np.ndarray]) the batch of actions
        :param reward: (Union[Tuple[float], np.ndarray]) the batch of the rewards of the transition
        :param obs_tp1: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the current batch of observations
        :param done: (Union[Tuple[bool], np.ndarray]) terminal status of the batch

        Note: uses the same names as .add to keep compatibility with named argument passing
            but expects iterables and arrays with more than 1 dimensions
        """
        idx = self._next_idx
        super().extend(obs_t, action, reward, obs_tp1, done)
        while idx != self._next_idx:
            self._it_sum[idx] = self._max_priority ** self._alpha
            self._it_min[idx] = self._max_priority ** self._alpha
            idx = (idx + 1) % self._maxsize

    def _sample_proportional(self, batch_size):
        mass = []
        total = self._it_sum.sum(0, len(self._storage) - 1)
        # TODO(szymon): should we ensure no repeats?
        mass = np.random.random(size=batch_size) * total
        idx = self._it_sum.find_prefixsum_idx(mass)
        return idx

    def sample(self, batch_size, beta=0):
        """
        Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.

        :param batch_size: (int) How many transitions to sample.
        :param beta: (float) To what degree to use importance weights (0 - no corrections, 1 - full correction)
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
            - weights: (numpy float) Array of shape (batch_size,) and dtype np.float32 denoting importance weight of
                each sampled transition
            - idxes: (numpy int) Array of shape (batch_size,) and dtype np.int32 idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)
        p_sample = self._it_sum[idxes] / self._it_sum.sum()
        weights = (p_sample * len(self._storage)) ** (-beta) / max_weight
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """
        Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        assert np.min(priorities) > 0
        assert np.min(idxes) >= 0
        assert np.max(idxes) < len(self.storage)
        self._it_sum[idxes] = priorities ** self._alpha
        self._it_min[idxes] = priorities ** self._alpha

        self._max_priority = max(self._max_priority, np.max(priorities))

