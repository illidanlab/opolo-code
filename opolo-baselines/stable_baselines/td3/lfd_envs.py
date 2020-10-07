from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import gym
import numpy as np


class AbsorbingWrapper(gym.ObservationWrapper):
  """Wraps an environment to have an indicator dimension.
  The indicator dimension is used to represent absorbing states of MDP.
  If the last dimension is 0. It corresponds to a normal state of the MDP,
  1 corresponds to an absorbing state.
  The environment itself returns only normal states, absorbing states are added
  later.
  This wrapper is used mainly for GAIL, since we need to have explicit
  absorbing states in order to be able to assign rewards.
  """

  def __init__(self, env):
    super(AbsorbingWrapper, self).__init__(env)
    obs_space = self.observation_space
    self.observation_space = gym.spaces.Box(
        shape=(obs_space.shape[0] + 1,),
        low=obs_space.low[0],
        high=obs_space.high[0])
    self.max_episode_steps = self.spec.max_episode_steps 

  def observation(self, observation):
    return self.get_non_absorbing_state(observation)

  def get_non_absorbing_state(self, obs):
    """Converts an original state of the environment into a non-absorbing state.
    Args:
      obs: a numpy array that corresponds to a state of unwrapped environment.
    Returns:
      A numpy array corresponding to a non-absorbing state obtained from input.
    """
    return np.concatenate([obs, [0]], -1)

  def get_absorbing_state(self):
    """Returns an absorbing state that corresponds to the environment.
    Returns:
      A numpy array that corresponds to an absorbing state.
    """
    obs = np.zeros(self.observation_space.shape)
    obs[-1] = 1
    return obs
