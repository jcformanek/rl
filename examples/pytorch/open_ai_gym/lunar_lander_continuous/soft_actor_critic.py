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


def create_mlp(layer_dims, final_relu=False):
        """
        layer_dims must be a list of int.
        """
        layers = []

        for i in range(len(layer_dims)-1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i != len(layer_dims)-2 or final_relu:
                layers.append(nn.ReLU())

        return nn.Sequential(*layers)

LOG_STD_MIN = -20
LOG_STD_MAX = 2

class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, act_limit, hidden_dims):
        super().__init__()

        self.mlp = create_mlp([obs_dim] + hidden_dims, final_relu=True)
        self.mu_layer = nn.Linear(hidden_dims[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], act_dim)
        self.act_limit = act_limit


    def forward(self, obs, deterministic=False, with_logprob=True):
        mlp_out = self.mlp(obs)
        mu = self.mu_layer(mlp_out)
        log_std = self.log_std_layer(mlp_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)

        # used for test time, return the mean action
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample() #rsample has something to do with the reparametrization trick.

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1) # HERE should dim be axis?? and should it be -1??
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action) #squashing 
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_dims):
        super().__init__()
        self.q = create_mlp([obs_dim+act_dim]+hidden_dims + [1])

    def forward(self, obs, act):
        # give action as input instead of having an output for each action
        inp = torch.cat([obs, act], dim=-1) #does dim=1 work?
        q = self.q(inp)
        return torch.squeeze(q, -1) #does dim=1 work?


class MLPActorCritic(nn.Module):
    
    def __init__(self, obs_dim, act_dim, act_limit, hidden_dims):
        super().__init__()

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, act_limit, hidden_dims)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_dims)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_dims)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
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


class SACAgent:

    def __init__(self, obs_dim, act_dim, act_limit, 
        hidden_dims=[256, 128], replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100):

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.gamma = gamma
        self.batch_size = batch_size
        self.polyak = polyak
        self.alpha = alpha

        self.ac = MLPActorCritic(obs_dim, act_dim, act_limit, hidden_dims=hidden_dims)
        self.ac_target = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for param in self.ac_target.parameters():
            param.requires_grad = False

        self.replay_buffer = ReplayBuffer(obs_dim, act_dim, replay_size)

        # Policy optimizer
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr)

        # Q function optimizer
        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.q_optimizer = Adam(self.q_params, lr=lr)

    def compute_loss_q(self, data):
        obs, act, rew, next_obs, done = data["obs"], data["act"], data["rew"], data["next_obs"], data["done"]
        q1 = self.ac.q1(obs, act)
        q2 = self.ac.q2(obs, act)

        with torch.no_grad():
            # target actions come from *current* policy
            a_target, logp_a_target = self.ac.pi(next_obs)

            #Target Q-values
            q1_pi_target = self.ac_target.q1(next_obs, a_target)
            q2_pi_target = self.ac_target.q2(next_obs, a_target)
            q_pi_target = torch.min(q1_pi_target, q2_pi_target)
            backup = rew + self.gamma * (1-done) * (q_pi_target - self.alpha * logp_a_target)

        # MSE loss against Max Entropy Bellman backup
        loss_q1 =((q1 - backup)**2).mean()
        loss_q2 =((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q

    def compute_loss_pi(self, data):
        obs = data['obs']
        pi, logp_pi = self.ac.pi(obs)
        q1_pi = self.ac.q1(obs, pi)
        q2_pi = self.ac.q2(obs, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        return loss_pi

    def choose_action(self, obs, deterministic=False):
        return self.ac.act(torch.as_tensor(obs, dtype=torch.float32), deterministic=deterministic)

    def learn(self):
        
        batch = self.replay_buffer.sample_batch(batch_size=self.batch_size)
        # first run one gradient descent step on Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(batch)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks parameters so you dont waste computational effort 
        # computing gradients for them during the policy learning step.
        for param in self.q_params:
            param.requires_grad = False

        # Run one gradient decent step for pi
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(batch)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-Network parameters
        for param in self.q_params:
            param.requires_grad = True

        # Finally update target networks via polyak averaging
        with torch.no_grad():
            for p, p_target in zip(self.ac.parameters(), self.ac_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_target.data.mul_(self.polyak)
                p_target.data.add_((1-self.polyak) * p.data)

        
def train_sac_agent(agent, env, steps_per_epoch=4000, epochs=100, start_steps=10000,
    update_after=1000, update_every=50, max_ep_len=1000):

    # prepare for interaction with the environment 
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    ep_lens, ep_rets, avg_ep_rets = [], [], []
    obs, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experinece in env and update each epoch
    for t in range(total_steps):

        # until start_steps have elapsed, randomly sample actions
        # for beeter exploration. Afterwards use the learned policy
        if t > start_steps:
            act = agent.choose_action(obs) 
        else:
            act = env.action_space.sample()

        # step the env
        next_obs, rew, done, _ = env.step(act)
        ep_ret += rew
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        done = False if ep_len == max_ep_len else done

        agent.replay_buffer.store(obs, act, rew, next_obs, done)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        obs = next_obs

        if done or (ep_len == max_ep_len):
            ep_lens.append(ep_len)
            ep_rets.append(ep_ret)
            avg_ep_ret = np.mean(ep_rets[-100:])
            avg_ep_rets.append(avg_ep_ret)
            obs, ep_ret, ep_len = env.reset(), 0, 0

        # Handle learning
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
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
    agent = SACAgent(obs_dim, act_dim, act_limit)

    # Train Agent
    avg_ep_rets = train_sac_agent(agent, env, epochs=1)

    # Plot Average Returns During Training
    plt.plot(np.arange(len(avg_ep_rets)), avg_ep_rets)
    plt.show()