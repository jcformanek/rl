"""
This code was closely adapted from OpenAi's SpinningUp tutorial, which can be accessed at 
https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo

"""
import gym
from gym.spaces import Box, Discrete

import numpy as np
import scipy.signal

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.categorical import Categorical


def make_env(env_name):
    if env_name not in ['CartPole-v1', 'LunarLander-v2']:
        raise ValueError("This PPO implementation has not been tested on this gym environment. Please chose a valid environment.")
    else:
        env = gym.make(env_name)
        return env


def mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        linear_layer = nn.Linear(sizes[i], sizes[i + 1])
        layers.append(linear_layer)

        if i < len(sizes) - 2:
            activation = nn.ReLU()
            layers.append(activation)

    return nn.Sequential(*layers)


class MLPCategoricalActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes):
            """
            obs_dim :: int
            act_dim :: int
            hidden_sizes :: (int, int)
            """
            super().__init__()
            sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
            self.logits_net = mlp(sizes)

    def _distribution(self, input):
        """
        input :: torch tensor 
        """
        logits = self.logits_net(input)
        distribution = Categorical(logits=logits)
        return distribution

    def _log_prob_from_distribution(self, distribution, idx):
        """
        distribution :: torch distribution Catagorical
        idx :: int
        """
        return distribution.log_prob(idx)

    def forward(self, obs, act=None):
        """
        obs :: torch tensor
        act :: int
        """
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes):
        """
        obs_dim :: int
        hidden_sizes :: (int, int)
        """
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1])

    def forward(self, obs):
        """
        obs :: torch tensor
        """
        values = self.v_net(obs)
        values = torch.squeeze(values, -1) # reshape values into 1D array
        return values


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(64, 64)):
        """
        observation_space :: gym.env.observation_space
        action_space :: gym.env.action_space
        """"""
        obs :: torch tensor
        """
        super().__init__()

        obs_dim = observation_space.shape[0]

        # Policy Network
        if isinstance(action_space, Box):
            raise NotImplementedError("Havnt implemented Gaussian Actor yet")
            # act_dim = action_space.shape[0]
            # self.pi = MLPGaussianActor(obs_dim, act_dim, hidden_sizes)
        elif isinstance(action_space, Discrete):
            act_dim = action_space.n
            self.pi = MLPCategoricalActor(obs_dim, act_dim, hidden_sizes)

        # Value Network
        self.v = MLPCritic(obs_dim, hidden_sizes)

    def step(self, obs):
        """""
        obs :: torch tensor
        """
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        """
        obs :: torch tensor
        """
        return self.step(obs)[0]


class Buffer:

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):

        def combine_shape(size, shape):
            return (size, shape) if np.isscalar(shape) else (size, *shape)

        self.obs_buf = np.zeros(combine_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def _discount_cumsum(self, x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size # buffer should still have space
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = self._discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr


    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size # buffer must be full

        # Next two lines implement advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                adv=self.adv_buf, logp=self.logp_buf)

        self.ptr, self.path_start_idx = 0, 0

        return {k: torch.as_tensor(np.ascontiguousarray(v), dtype=torch.float32) for k,v in data.items()}


class PPOAgent:

    def __init__(self, observation_space, action_space, steps_per_epoch=4000, hidden_sizes=(64, 64), gamma=0.99, lam=0.97, 
        clip_ratio=0.2, pi_lr=3e-4, vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, max_ep_len=1000):

        self.clip_ratio = clip_ratio
        self.steps_per_epoch = steps_per_epoch
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.max_ep_len = max_ep_len

        self.ac = MLPActorCritic(env.observation_space, env.action_space, hidden_sizes=hidden_sizes)

        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)

        obs_dim = observation_space.shape
        if isinstance(action_space, Box):
            act_dim = action_space.shape[0]
        elif isinstance(action_space, Discrete):
            act_dim = action_space.n 

        self.buf = Buffer(obs_dim, act_dim, steps_per_epoch, gamma=gamma, lam=lam)

    # Function for computing PPO policy loss
    def _compute_loss_pi(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        pi, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        return loss_pi

    def _compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']
        return ((self.ac.v(obs) - ret)**2).mean()

    def _learn(self):
        data = self.buf.get()
        pi_l_old = self._compute_loss_pi(data).item()
        v_l_old = self._compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi = self._compute_loss_pi(data)
            ### TO DO implement KL early stopping
            loss_pi.backward()
            self.pi_optimizer.step()

        # Value function learning
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self._compute_loss_v(data)
            loss_v.backward()
            self.vf_optimizer.step()

    def train(self, env, epochs):
        o, ep_ret, ep_len = env.reset(), 0, 0
        ep_rets = []
        for epoch in range(epochs):
            for t in range(self.steps_per_epoch):
                a, v, logp = self.ac.step(torch.as_tensor(np.ascontiguousarray(o), dtype=torch.float32))
                next_o, r, d, _ = env.step(a)
                ep_ret += r
                ep_len += 1

                self.buf.store(o, a, r, v, logp)

                o = next_o

                timeout = ep_len == self.max_ep_len
                terminal = d or timeout
                epoch_ended = t == self.steps_per_epoch-1

                if terminal or epoch_ended:
                    if epoch_ended and not(terminal):
                        pass
                        # print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        _, v, _ = self.ac.step(torch.as_tensor(np.ascontiguousarray(o), dtype=torch.float32))
                    else:
                        v = 0

                    self.buf.finish_path(v)

                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        #logger.store(EpRet=ep_ret, EpLen=ep_len)
                        ep_rets.append(ep_ret)

                    o, ep_ret, ep_len = env.reset(), 0, 0

            self._learn()

            print("Epoch:", epoch, "Avg. Return:", np.mean(ep_rets[-100:]))

if __name__=="__main__":
    env = make_env("LunarLander-v2")
    agent = PPOAgent(env.observation_space, env.action_space)
    agent.train(env, 100)
    env.close()
