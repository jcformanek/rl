import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam

import numpy as np
import gym

def discount_cumsum(x, discount):
    y = [0 for i in range(len(x))]
    for i in range(len(x)-1, -1, -1):
        if i == len(x)-1:
            y[i] = x[i]
        else:
            y[i] = x[i] + discount * y[i+1]
    return y


class Buffer:

    def __init__(self, size, gamma=0.99, lam=0.97):
        super().__init__()
        self.buf = []
        self.ret_buf = []
        self.adv_buf = []
        self.size = size
        self.start_of_trajectory = 0
        self.ctr = 0
        self.gamma = gamma
        self.lam = lam

    def store_transition(self, obs, act, rew, val, logp):
        assert self.ctr < self.size
        self.buf.append((obs, act, rew, val, logp))
        self.ctr += 1

    def end_of_trajectory(self, last_val=0):
        path_slice = slice(self.start_of_trajectory, self.ctr)
        _, _, rew_slice, val_slice, _ = zip(*self.buf[path_slice])
        rew_slice = list(rew_slice)
        rew_slice.append(last_val)
        rew_slice = np.array(rew_slice)
        val_slice = list(val_slice)
        val_slice.append(last_val)
        val_slice = np.array(val_slice)

        # Compute GAE
        deltas = rew_slice[:-1] + self.gamma * val_slice[1:] - val_slice[:-1]
        adv_slice = discount_cumsum(deltas, self.gamma * self.lam)
        self.adv_buf += adv_slice

        # Store returns-to-go
        ret_slice = discount_cumsum(rew_slice, self.gamma)[:-1]
        self.ret_buf += ret_slice

        self.start_of_trajectory = self.ctr

    def get(self):
        assert self.ctr == self.size
        obs_batch, act_batch, rew_batch, val_batch, logp_batch = zip(*self.buf)
        adv_batch, ret_batch = self.adv_buf, self.ret_buf
        data = {'obs': obs_batch, 'act': act_batch, 'rew': rew_batch, 'val': val_batch,
            'logp': logp_batch, 'ret': ret_batch, 'adv': adv_batch}

        # Convert to tensors
        data = {k: torch.tensor(np.array(v, dtype=np.float32)) for k,v in data.items()}

        # Normalize advantages
        adv_mean, adv_std = torch.mean(data['adv']), torch.std(data['adv'])
        data['adv'] = (data['adv'] - adv_mean) / adv_std

        self.buf = []
        self.adv_buf = []
        self.ret_buf = []

        self.ctr, self.start_of_trajectory = 0, 0

        return data


class Actor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 128)):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dims = hidden_dims
        self.logits_net = self._create_mlp()
        

    def _create_mlp(self):
        layers = []

        # Input layer
        input_layer = nn.Linear(self.obs_dim, self.hidden_dims[0])
        layers.append(input_layer)
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(self.hidden_dims)-1):
            layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
            layers.append(nn.ReLU())

        # Output layer dim == act_dim
        output_layer = nn.Linear(self.hidden_dims[-1], act_dim)
        layers.append(output_layer)

        return nn.Sequential(*layers)



    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class Critic(nn.Module):

    def __init__(self, obs_dim, hidden_dims=(256, 128)):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dims = hidden_dims
        self.v_net = self._create_mlp()

    def _create_mlp(self):
        layers = []

        # Input layer
        input_layer = nn.Linear(self.obs_dim, self.hidden_dims[0])
        layers.append(input_layer)
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(self.hidden_dims)-1):
            layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
            layers.append(nn.ReLU())

        # Output layer dim == 1
        output_layer = nn.Linear(self.hidden_dims[-1], 1)
        layers.append(output_layer)

        return nn.Sequential(*layers)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 128)):
        super().__init__()

        # build policy function
        self.pi = Actor(obs_dim, act_dim, hidden_dims=hidden_dims)

        # build value function
        self.v  = Critic(obs_dim, hidden_dims=hidden_dims)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


def train_actor_critic(env, ac, steps_per_epoch=256, epochs=50, max_ep_len=1000, gamma=0.99, lam=0.97, 
    pi_lr=1e-3, vf_lr=1e-3, train_v_iters=80):

    buf = Buffer(steps_per_epoch,  gamma=gamma, lam=lam)

    # Set up function for computing policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        loss_pi = -(logp * adv).mean()

        return loss_pi

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    def learn():
        data = buf.get()

        # Train policy with a single step of gradient descent
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            vf_optimizer.step()

    num_games = 0
    ep_rets, ep_lens = [], []
    avg_ep_rets, avg_ep_lens = [], []
    obs, ep_ret, ep_len = env.reset(), 0, 0
    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            act, val, logp = ac.step(torch.as_tensor(obs, dtype=torch.float32))

            next_obs, rew, done, _ = env.step(act)
            ep_ret += rew
            ep_len += 1

            # Store transition
            buf.store_transition(obs, act, rew, val, logp)
            
            # Update obs 
            obs = next_obs

            timeout = ep_len == max_ep_len
            terminal = done or timeout
            epoch_ended = t == steps_per_epoch-1

            if terminal or epoch_ended:
                # if epoch_ended and not(terminal):
                #     print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)

                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, val, _ = ac.step(torch.as_tensor(obs, dtype=torch.float32))
                else:
                    val = 0

                buf.end_of_trajectory(val)

                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    ep_rets.append(ep_ret), ep_lens.append(ep_len)
                    avg_ep_rets.append(np.mean(ep_rets[-100:]))
                    avg_ep_lens.append(np.mean(ep_lens[-100:]))
                    num_games += 1

                obs, ep_ret, ep_len = env.reset(), 0, 0

        learn()

        if ((epoch + 1) % 10 == 0):
            print(f"Epoch: {epoch+1}    Num. games: {num_games}     Avg. score: {round(avg_ep_rets[-1], 1)}     Avg. ep len: {round(avg_ep_lens[-1], 1)}")


if __name__=='__main__':
    ENV_NAME = 'CartPole-v1'
    HIDDEN_DIMS = (100,)
    EPOCHS = 1000

    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    ac = ActorCritic(obs_dim, act_dim, hidden_dims=HIDDEN_DIMS)

    train_actor_critic(env, ac, epochs=EPOCHS)

    

