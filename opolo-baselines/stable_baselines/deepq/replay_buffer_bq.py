import random

import numpy as np

from stable_baselines.common.segment_tree import SumSegmentTree, MinSegmentTree


class DACBuffer(object):
    """
    Each observation has an extra dimension to indicate whether it is absorbing or not.
    """
    def __init__(self, max_size, max_episode_steps, env):
        self._storage = []
        self._next_idx = 0
        self._maxsize = max_size
        self._max_episode_steps = max_episode_steps
        self.env = env

    def add(self, obs_t, action, reward, obs_tp1, done, if_absorb, episode_score, current_steps):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        # determine if the next step is abosrbing
        if not done or current_steps == self._max_episode_steps:
            if_absorb = 0
        else:
            if_absorb = 1
        # replace next_obs to be absorbing if necessary
        if done and current_steps < self._max_episode_steps:
            obs_tp1 = self.env.get_absorbing_state()
        
        data = (obs_t, action, reward, obs_tp1, done, if_absorb, episode_score)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
    
    def reset(self):
        """
        free space, move pointers
        """
        del self._storage
        self._storage = []
        self._next_idx = 0
    
    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones, demos, scores = [], [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, demo, score = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            demos.append(demo)
            scores.append(score)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones).reshape(-1,1), np.array(demos).reshape(-1,1), np.array(scores).reshape(-1,1)

    def get_episode(self, **_kwargs):
        """
        get current episode
        """
        return [self._storage[i] for i in range(len(self._storage))]

class TrajectoryBuffer(object):
    def __init__(self, max_size, gamma=0.99):
        self._storage = []
        self._next_idx = 0
        self._maxsize = max_size
        self.gamma = gamma

    def add(self, obs_t, action, reward, obs_tp1, done, demo, episode_score, true_reward):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        data = (obs_t, action, reward, obs_tp1, done, demo, episode_score, true_reward)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = int((self._next_idx + 1) % self._maxsize)
    
    def reset(self):
        """
        free space, move pointers
        """
        del self._storage
        self._storage = []
        self._next_idx = 0
    
    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones, demos, scores, true_rewards = [], [], [], [], [], [], [], [] 
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, demo, score, true_reward = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            demos.append(demo)
            scores.append(score)
            true_rewards.append(true_reward)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones).reshape(-1,1), np.array(demos).reshape(-1,1), np.array(scores).reshape(-1,1), np.array(true_rewards).reshape(-1,1) 

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
    
    def get_episode_return(self, es):
        ret = []
        episode_len = len(self._storage)
        for i in range(episode_len):
            data = self._storage[i] 
            discount_return = (self.gamma ** (episode_len-i-1)) * es 
            data = data + (discount_return,)
            ret.append(data)
        return ret
        


class ReplayBufferWithDemo(object):
    def __init__(self, size, demo_size):
        """
        Implements a ring buffer (FIFO).

        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            self-generated transitions are dropped.
        :param demo_size: (int) Number of expert demonstration transitions to store in the buffer. They are fixed, and once stored, will never be replaced with new transitions. --Judy
        """
        self._storage = []
        self._maxsize = size
        self._demo_size = demo_size
        self._next_demo_idx = 0
        self._next_self_idx = demo_size

    def __len__(self):
        return len(self._storage)

    @property
    def storage(self):
        """[(np.ndarray, float, float, np.ndarray, bool)]: content of the replay buffer"""
        return self._storage

    @property
    def buffer_size(self):
        """float: Max capacity of the buffer"""
        return self._maxsize

    @property
    def demo_buffer_size(self):
        """float: Capacity of the demo buffer"""
        return self._demo_size

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

    def add_demo_data(self, obs_t, action, reward, obs_tp1, done, demo, p):
        """
        similar to add, but here we only add expert demonstration data.
        we use another pointer: _next_demo_idx to replace old demo-data. 
        """
        assert demo == 1
        data = (obs_t, action, reward, obs_tp1, done, demo, p) 
        ## put data in the replay buffer
        if self._next_demo_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_demo_idx] = data
        self._next_demo_idx = (self._next_demo_idx + 1) % self._demo_size
    

    def add_self_data(self, obs_t, action, reward, obs_tp1, done, demo, p):
        """
        add self-generated data to the buffer. 
        we use another pointer: _next_self_idx to replace old self-generated-data. 
        - The data are stored in such a way that the expert demonstrations
        are always stored first.
        - This function will not be called until demo-buffer is full.
        """ 
    
        data = (obs_t, action, reward, obs_tp1, done, demo, p)
        assert len(self) >= self._demo_size and demo == 0

        ## put data in the replay buffer
        if self._next_self_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_self_idx] = data

        ## when replacing data, it will only replace self-generated transitions 
        ##      ------------- ......... | -----------
        ##      demo - data   ......... | self- data
        ## idx: 0 ........    ......... | self._demo_size, self._demo_size + 1, ........
           
        self._next_self_idx = (self._next_self_idx + 1 - self._demo_size ) % (self._maxsize - self._demo_size) + self._demo_size 

    def add_by_overlap_self_data(self, obs_t, action, reward, obs_tp1, done, demo, p):
        """
        subsititute self-data by overlapping it with expert demo data and shrink self-data size.
        The data are stored in such a way that the expert demonstrations
        are always stored first.
        This function will not be called until demo-buffer is full.
        """
        assert demo == 1 and len(self) >= self._demo_size
        data = (obs_t, action, reward, obs_tp1, done, demo, p)
        self._demo_size -= 1
        self._storage[self._demo_size] = data
        self._next_demo_idx = self._next_demo_idx % self._demo_size

    def add_by_overlap_demo(self, obs_t, action, reward, obs_tp1, done, demo, p):
        """
        subsititute demo data by overlapping it with self-generated data and shrink demo size.
        The data are stored in such a way that the expert demonstrations
        are always stored first.
        This function will not be called until demo-buffer is full.
        """
        assert demo == 0 and len(self) >= self._demo_size
        data = (obs_t, action, reward, obs_tp1, done, demo, p)
        self._demo_size -= 1
        self._storage[self._demo_size] = data
        self._next_demo_idx = self._next_demo_idx % self._demo_size
        
    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones, demos, ps = [], [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, demo, p = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            demos.append(demo)
            ps.append(p)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), np.array(demos), np.array(ps)

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
    
    def sample_demo_data(self, batch_size, **_kwargs):
        """
        sample only demonstration data
        """
        curr_demo_size = min(self._demo_size, len(self))
        idxes = [random.randint(0, curr_demo_size - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def sample_self_data(self, batch_size, **_kwargs):
        """
        sample only self-generated data
        """
        assert len(self) > self._demo_size + 0.1 * batch_size
        curr_start_idx = self._demo_size 
        curr_end_idx = len(self) - 1
        idxes = [random.randint(curr_start_idx, curr_end_idx) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class ReplayBufferExtend(object):
    def __init__(self, size):
        """
        Implements a ring buffer (FIFO).

        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        
        each sample in the buffer contains (s, a, r, s', demo(1 or 0), p(probability output from the discriminator))
        -- Judy
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    @property
    def storage(self):
        """[(np.ndarray, float, float, np.ndarray, bool)]: content of the replay buffer"""
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

    def add(self, obs_t, action, reward, obs_tp1, done, demo, es, true_reward):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition, might be shaped reward
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        :param es: episodic score (normed and discounted)
        :param true_reward: environment reward
        """
        data = (obs_t, action, reward, obs_tp1, done, demo, es, true_reward)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = int((self._next_idx + 1) % self._maxsize)

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones, demos, epi_socres, true_rewards = [], [], [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, demo, es, true_reward = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            demos.append(demo)
            epi_socres.append(es)
            true_rewards.append(true_reward)
        return np.array(obses_t), np.array(actions), np.array(rewards).reshape(-1,1), np.array(obses_tp1), np.array(dones).reshape(-1,1), np.array(demos).reshape(-1,1), np.array(epi_socres).reshape(-1,1), np.array(true_rewards).reshape(-1,1) 


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
        #print(len(self._storage))
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def encode_sample(self, idxes):
        return self._encode_sample(idxes)


class ExpertBuffer(object):
    """
    Trajectory-wise prioritized buffer.
    Each trajectory carries a priority (normalized score)
    new trajectory will be added if len(self.episodes) < self.max_episode_n
    otherwise, replace old trajectories with the lowest score.
    """
    def __init__(self, max_episode_n):
        assert max_episode_n > 0
        self.episodes = []
        self.episode_scores = [] # normalized
        self.min_score = float('inf')
        self.max_score = -float('inf') 
        self.max_episode_n = max_episode_n
        self.current_episode_idx = 0
    
    def get_queue_slot(self, es):
        if len(self.episodes) < self.max_episode_n:
            self.episodes.append([]) # create new queue
            self.current_episode_idx = len(self.episodes) - 1 
            self.episode_scores.append(es)
        else:
            # replace old trajectory with lowest score
            episode_idx = self.episode_scores.index(self.min_score)
            old_episode = self.episodes[episode_idx]
            self.episodes[episode_idx] = [] # reset queue
            del old_episode # free space 
            self.episode_scores[episode_idx] = es
            self.current_episode_idx = episode_idx 

        self.min_score = min(self.episode_scores)
        self.max_score = max(self.episode_scores)
        return #self.current_episode_idx, self.current_sample_idx 


    def add_with_priority(self, obs_t, action, reward, obs_tp1, done, demo, priority):
        self.add(obs_t, action, reward, obs_tp1, done, demo, priority)

    def add(self, obs_t, action, reward, obs_tp1, done, demo, p):
        data = (obs_t, action, reward, obs_tp1, done, demo, p)
        self.episodes[self.current_episode_idx].append(data)

    def update_priorities(self, idxes, priorities):
        pass

    def _encode_sample(self, episode_idxes, idxes):
        obses_t, actions, rewards, obses_tp1, dones, demos, scores = [], [], [], [], [], [], []
        for idx, i in enumerate(idxes):
            episode_idx = episode_idxes[idx]
            data = self.episodes[episode_idx][i]
            obs_t, action, reward, obs_tp1, done, demo, score = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            demos.append(demo)
            scores.append(score)
        return np.array(obses_t), np.array(actions), np.array(rewards).reshape(-1,1), np.array(obses_tp1), np.array(dones).reshape(-1,1), np.array(demos).reshape(-1,1), np.array(scores).reshape(-1,1)

    def sample_uniform(self, batch_size):
        episode_idxes = np.random.choice(len(self.episode_scores), size=batch_size)
        idxes = [] 
        for i in range(batch_size):
            epi_idx = episode_idxes[i]#random.randint(0, len(self.episode_scores)-1)
            idx = random.randint(0, len(self.episodes[epi_idx])-1)
            idxes.append(idx)
        return self._encode_sample(episode_idxes, idxes) 

    def sample(self, batch_size, beta=0):
        assert self.min_score > 0
        normalized_p = np.array(self.episode_scores)/np.sum(self.episode_scores)
        episode_idxes = np.random.choice(len(self.episode_scores), size=batch_size, p=normalized_p)
        idxes = [] 
        for i in range(batch_size):
            epi_idx = episode_idxes[i]#random.randint(0, len(self.episode_scores)-1)
            idx = random.randint(0, len(self.episodes[epi_idx])-1)
            idxes.append(idx)
        return self._encode_sample(episode_idxes, idxes) + (np.ones((batch_size,1)) ,[1 for _ in range(batch_size)])
        


class PrioritizedReplayBufferExtend(ReplayBufferExtend):
    def __init__(self, size, alpha):
        """
        Create Prioritized Replay buffer.

        See Also ReplayBuffer.__init__

        :param size: (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories
            are dropped.
        :param alpha: (float) how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        super(PrioritizedReplayBufferExtend, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add_with_priority(self, obs_t, action, reward, obs_tp1, done, demo, priority, true_reward):
        idx = self._next_idx
        super().add(obs_t, action, reward, obs_tp1, done, demo, priority, true_reward)
        assert priority> 0
        self._it_sum[idx] = priority ** self._alpha
        self._it_min[idx] = priority ** self._alpha
        self._max_priority = max(self._max_priority, priority)

    def add(self, obs_t, action, reward, obs_tp1, done, demo, p, true_reward):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        :param demo: (bool) is the transition a demonstration sample
        :param p: (float) probability output of the discriminator
        """
        idx = self._next_idx
        super().add(obs_t, action, reward, obs_tp1, done, demo, p, true_reward)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample_uniform(self, batch_size):
        return super().sample(batch_size) 

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

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights).reshape(-1,1)
        encoded_sample = self._encode_sample(idxes)
        #idxes = np.array(idxes).reshape(-1,1)
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
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)



class ReplayBuffer(object):
    def __init__(self, size):
        """
        Implements a ring buffer (FIFO).

        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    @property
    def storage(self):
        """[(np.ndarray, float, float, np.ndarray, bool)]: content of the replay buffer"""
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

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        data = (obs_t, action, reward, obs_tp1, done)

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
        return self._encode_sample(idxes)


class PrioritizedReplayBufferWithDemo(ReplayBufferWithDemo):
    def __init__(self, size, demo_size, alpha):
        """
        Create Prioritized Replay buffer.

        See Also ReplayBufferWithDemo.__init__

        :param size: (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories
            are dropped.
        :param demo_size: (int) Number of expert demonstrations to store in the buffer. They will never be dropped.
        :param alpha: (float) how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        super(PrioritizedReplayBufferWithDemo, self).__init__(size, demo_size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    #def add(self, obs_t, action, reward, obs_tp1, done, demo, p):
    #    """
    #    add a new transition to the buffer

    #    :param obs_t: (Any) the last observation
    #    :param action: ([float]) the action
    #    :param reward: (float) the reward of the transition
    #    :param obs_tp1: (Any) the current observation
    #    :param done: (bool) is the episode done
    #    :param demo: (bool) is the transition an expert demonstration

    #    """
    #    idx = self._next_idx
    #    super().add(obs_t, action, reward, obs_tp1, done, demo, p)
    #    self._it_sum[idx] = self._max_priority ** self._alpha
    #    self._it_min[idx] = self._max_priority ** self._alpha
    
    def add_demo_data(self, obs_t, action, reward, obs_tp1, done, demo, p): 
        
        idx = self._next_demo_idx
        super().add_demo_data(obs_t, action, reward, obs_tp1, done, demo, p)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def add_self_data(self, obs_t, action, reward, obs_tp1, done, demo, p):
        idx = self._next_self_idx
        super().add_self_data(obs_t, action, reward, obs_tp1, done, demo, p)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha
    
    def add_by_overlap_demo(self, obs_t, action, reward, obs_tp1, done, demo, p):
        idx = self._demo_size - 1
        super().add_by_overlap_demo(obs_t, action, reward, obs_tp1, done, demo, p)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

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
            - demo_mask: (numpy bool) demo_mask[i] = 1 if it is an expert demonstration 
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

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights).reshape(-1,1)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, np.array(idxes).reshape(-1,1)])

    def sample_demo_data(self, batch_size, **_kwargs):
        """
        sample only demonstration data, used to train discriminator
        """
        curr_demo_size = min(self._demo_size, len(self))
        idxes = [random.randint(0, curr_demo_size - 1) for _ in range(batch_size)]
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [None, idxes])

    def sample_self_data(self, batch_size, **_kwargs):
        """
        sample only self-generated data, used to train discriminator
        """
        assert len(self) > self._demo_size + 0.1 * batch_size
        curr_start_idx = self._demo_size 
        curr_end_idx = len(self) - 1
        idxes = [random.randint(curr_start_idx, curr_end_idx) for _ in range(batch_size)]
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [None, idxes])

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
        for idx, priority in zip(idxes, priorities): 
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha
            self._max_priority = max(self._max_priority, priority)



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

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res


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

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
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
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)
