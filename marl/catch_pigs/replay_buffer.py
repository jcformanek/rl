import numpy as np

class ReplayBuffer():
    def __init__(self, size, obs_dims):
        """
        size :: int
        obs_dims :: (int, ... , int)
        """
        self.size = size
        self.ctr = 0
        self.obs_buffer = np.zeros((size, *obs_dims))
        self._obs_buffer = np.zeros((size, *obs_dims))
        self.act_buffer = np.zeros(size)
        self.rew_buffer = np.zeros(size)
        self.done_buffer = np.zeros(size)


    def push(self, obs, act, rew, _obs, done):
        idx = self.ctr % self.size
        self.obs_buffer[idx] = obs
        self.act_buffer[idx] = act
        self.rew_buffer[idx] = rew
        self._obs_buffer[idx] = _obs
        self.done_buffer[idx] = done
        self.ctr += 1


    def sample(self, batch_size):
        """
        TO DO: Maybe redesign to return non repeating indxs!!
        """
        hi = min(self.size, self.ctr)
        idxs = np.random.random_integers(0, hi, batch_size)
        return self.obs_buffer[idxs], self.act_buffer[idxs], self._obs_buffer[idxs], self.rew_buffer[idxs], self.done_buffer[idxs]