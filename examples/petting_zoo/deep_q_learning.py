import numpy as np
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Buffer:
    def __init__(self, size, obs_dim, act_dim):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.long)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.bool)
        self.size = size
        self.ctr = 0

    def store(self, obs, act, rew, next_obs, done):
        ptr = self.ctr % self.size

        self.obs_buf[ptr] = obs
        self.act_buf[ptr] = act
        self.rew_buf[ptr] = rew
        self.next_obs_buf[ptr] = next_obs
        self.done_buf[ptr] = done

        self.ctr += 1

    def sample(self, batch_size):
        max_idx = min(self.size, self.ctr)
        assert max_idx >= batch_size

        idxs = np.random.choice(max_idx, batch_size, replace=False)

        obs_batch = self.obs_buf[idxs]
        act_batch = self.act_buf[idxs]
        rew_batch = self.rew_buf[idxs]
        next_obs_batch = self.next_obs_buf[idxs]
        done_batch = self.done_buf[idxs]

        data = dict(obs=obs_batch, act=act_batch, rew=rew_batch, next_obs=next_obs_batch, done=done_batch)
        data = {k: torch.tensor(v) for k, v in data.items()}

        return data


class MLPDeepQNetwork(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_dims=[64, 64], lr=1e-3):
        super().__init__()
        layers = []

        # Input layer
        input_layer = nn.Linear(obs_dim, hidden_dims[0])
        layers.append(input_layer)
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())

        # Output layer
        output_layer = nn.Linear(hidden_dims[-1], act_dim)
        layers.append(output_layer)

        self.mlp = nn.Sequential(*layers)

    def forward(self, obs):
        return self.mlp(obs)


class Agent:
    def __init__(self, obs_dim, act_dim, hidden_dims=[256, 128], buf_size=10000, batch_size=256,
        lr=1e-3, target_replace=256, gamma=0.99, alpha=0.5, eps_min=0.01, eps_dec=1e-4):

        self.buf = Buffer(buf_size, obs_dim, act_dim)
        self.batch_size = batch_size

        self.q_eval = MLPDeepQNetwork(obs_dim, act_dim, hidden_dims=hidden_dims, lr=lr)
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_eval.parameters(), lr=lr)

        self.q_next = MLPDeepQNetwork(obs_dim, act_dim, hidden_dims=hidden_dims, lr=lr)
        self.q_next.eval()
        self.target_replace = target_replace

        self.gamma = gamma
        self.alpha = alpha

        self.action_space = [i for i in range(act_dim)]

        self.eps = 1
        self.eps_min = eps_min
        self.eps_dec = eps_dec

        self.learn_ctr = 0

    def choose_action(self, obs):
        if np.random.rand() < self.eps:
            act = np.random.choice(self.action_space)
        else:
            with torch.no_grad():
                obs = torch.tensor(obs, dtype=torch.float32)
                act = torch.argmax(self.q_eval(obs)).item()
        return act

    def store_transition(self, obs, act, rew, next_obs, done):
        self.buf.store(obs, act, rew, next_obs, done)

    def sample_memory(self):
        return self.buf.sample(self.batch_size)

    def replace_target(self):
        if self.learn_ctr % self.target_replace == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_eps(self):
        self.eps = self.eps - self.eps_dec if self.eps > self.eps_min else self.eps_min
        
    def learn(self):

        if self.buf.ctr < self.batch_size:
            return

        data = self.sample_memory()
        obs, act, rew, next_obs, done = data['obs'], data['act'], data['rew'], data['next_obs'], data['done']

        idxs = torch.tensor(np.arange(self.batch_size, dtype=np.long))

        q_eval = self.q_eval(obs)[idxs, act]

        with torch.no_grad():
            q_next = self.q_next(next_obs)
            q_target = rew + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_ctr += 1
        self.replace_target()
        self.decrement_eps()