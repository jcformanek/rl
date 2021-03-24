import numpy as np
from copy import deepcopy
import time
import gym
import itertools
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.optim import Adam
import torch.nn.functional as F
import matplotlib.pyplot as plt

def create_mlp(layer_dims, activation=nn.ReLU, final_activation=nn.Identity):
        """
        layer_dims must be a list of int.
        """
        layers = []

        for i in range(len(layer_dims)-1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i != len(layer_dims)- 2:
                layers.append(activation())
            else:
                layers.append(final_activation())

        return nn.Sequential(*layers)


class DeterministicMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_dims, act_limit):
        super().__init__()
        layer_dims = [obs_dim] + hidden_dims + [act_dim]
        self.pi = create_mlp(layer_dims, final_activation=nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        return self.act_limit * self.pi(obs)


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_dims):
        super().__init__()
        layer_dims = [obs_dim + act_dim] + hidden_dims + [1]
        self.q = create_mlp(layer_dims)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=1))
        return torch.squeeze(q, dim=1) # ensure q has the right shape


class MLPActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, act_limit, hidden_dims):
        super().__init__()
        self.pi = DeterministicMLPActor(obs_dim, act_dim, hidden_dims, act_limit)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_dims)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.ctr, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.ctr += 1

    def sample_batch(self, batch_size=32):
        max_idx = min(self.ctr, self.max_size)
        idxs = np.random.randint(0, max_idx, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs])
        return {k: torch.tensor(v, dtype=torch.float32) for k,v in batch.items()}




class DDPGAgent:

    def __init__(self, obs_dim, act_dim, act_limit, hidden_dims=[256,128], 
        replay_size=int(1e6), gamma=0.99, polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, act_noise=0.1):
        
        self.ac = MLPActorCritic(obs_dim, act_dim, act_limit, hidden_dims)
        self.ac_target = deepcopy(self.ac)

        # Freeze target network parameters
        for param in self.ac_target.parameters():
            param.requires_grad = False

        self.replay_buffer = ReplayBuffer(obs_dim, act_dim, replay_size)
        self.gamma = gamma
        self.polyak = polyak
        self.batch_size = batch_size
        self.noise_scale = act_noise
        self.act_limit = act_limit

        # Optimizers for the policy and Q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.ac.q.parameters(), lr=q_lr)

    # Compute DDPG Q-loss
    def compute_loss_q(self, data):
        obs, act, rew, next_obs, done = data['obs'], data['act'], data['rew'], data['next_obs'], data['done']
        
        q = self.ac.q(obs, act)

        # Bellman backup for Q function
        with torch.no_grad():
            pi_target = self.ac_target.pi(next_obs)
            q_pi_target = self.ac_target.q(next_obs, pi_target)
            backup = rew + self.gamma * (1 - done) * q_pi_target

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()

        return loss_q

    def compute_loss_pi(self, data):
        obs = data['obs']
        act = self.ac.pi(obs)
        q_pi = self.ac.q(obs, act)
        return -q_pi.mean()

    def choose_action(self, obs, noisy=True):
        act = self.ac.act(torch.as_tensor(obs, dtype=torch.float32))
        if noisy:
            act += self.noise_scale * np.random.randn(act_dim)
        return np.clip(act, -self.act_limit, self.act_limit)

    def learn(self):
        batch = self.replay_buffer.sample_batch(self.batch_size)

        #run one gradient descent step for Q
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(batch)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for param in self.ac.q.parameters():
            param.requires_grad = False

        # Run one gradient descent step for pi
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(batch)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-network
        for param in self.ac.q.parameters():
            param.requires_grad = True

        # Update target networks by polyak averaging
        with torch.no_grad():
            for param, param_target in zip(self.ac.parameters(), self.ac_target.parameters()):
                # we use in place tensor operations with the trailing _
                param_target.data.mul_(self.polyak)
                param_target.data.add_((1-self.polyak) * param.data)


def train_ddpg_agent(agent, env, epochs=100, steps_per_epoch=4000, start_steps=10000, 
    update_after=1000, update_every=50, max_ep_len=1000):

    total_steps = epochs * steps_per_epoch
    obs, ep_ret, ep_len = env.reset(), 0, 0
    ep_rets, avg_ep_rets = [], []

    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        if t > start_steps:
            act = agent.choose_action(obs)
        else:
            act = env.action_space.sample()

        # Step the environment
        next_obs, rew, done, _ = env.step(act)
        ep_ret += rew
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        done = False if ep_len==max_ep_len else done

        # Store transitions
        agent.replay_buffer.store(obs, act, rew, next_obs, done)

        # Super critical
        obs = next_obs

        # End of trajectory handling
        if done or (ep_len == max_ep_len):
            ep_rets.append(ep_ret)
            avg_ep_ret = np.mean(ep_rets[-100:])
            avg_ep_rets.append(avg_ep_ret)
            obs, ep_ret, ep_len = env.reset(), 0, 0

        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                agent.learn()

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            print(f"Epoch: {epoch}      Avg. Score: {avg_ep_rets[-1]}")
   
    return avg_ep_rets

if __name__=="__main__":

    # Create Environment
    env = gym.make("LunarLanderContinuous-v2")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    # Create Agent
    agent = DDPGAgent(obs_dim, act_dim, act_limit)

    # Train Agent
    avg_ep_rets = train_ddpg_agent(agent, env)

    # Plot Average Returns During Training
    plt.plot(np.arange(len(avg_ep_rets)), avg_ep_rets)
    plt.show()

