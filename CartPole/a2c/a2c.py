import os
import sys
import gym
import random
import numpy as np
from collections import namedtuple, deque

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class Memory():
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.masks = []

    def sample(self):
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        rewards = torch.Tensor(self.rewards)
        masks = torch.Tensor(self.masks)

        return states, actions, rewards, masks

    def store(self, state, action, reward, mask):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.masks.append(mask)


class ActorCriticNet(nn.Module):
    def __init__(self, obs_dims, num_actions):
        super(ActorCriticNet, self).__init__()
        self.obs_dim = obs_dims
        self.num_outputs = num_actions

        self.fc = nn.Linear(num_inputs, 32)
        self.fc_actor = nn.Linear(32, num_actions)
        self.fc_critic = nn.Linear(32, 1)

    def forward(self, input):
        x = F.relu(self.fc(input))
        policy = F.softmax(self.fc_actor(x))
        value = self.fc_critic(x)
        return policy, value

    def get_action(self, input):
        policy, _ = self.forward(input)
        policy = policy[0].data.numpy()

        action = np.random.choice(self.num_outputs, 1, p=policy)[0]
        return action

class A2C_Agent():
    def __init__(self, num_actions, obs_dim, lr=0.001, lambda_gae=0.96, gamma=0.99, 
                    ciritic_coefficient = 0.5, entropy_coefficient = 0.01):
        self.memory = memory
        self.num_actions = num_actions
        self.obs_dims = obs_dims
        self.net = ActorCriticNet(obs_dims, num_actions)
        self.optimizer = optim.Adam(net.parameters(), lr=lr)
        self.lambda_gae = lambda_gae
        self.gamma = gamma
        self.ciritic_coefficient = ciritic_coefficient
        self.entropy_coefficient = entropy_coefficient

    def get_gae(self, values, rewards, masks):
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        running_return = 0
        previous_value = 0
        running_advantage = 0

        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * masks[t]
            running_tderror = rewards[t] + self.gamma * previous_value * masks[t] - values.data[t]
            running_advantage = running_tderror + (self.gamma * self.lambda_gae) * running_advantage * masks[t]

            returns[t] = running_return
            previous_value = values.data[t]
            advantages[t] = running_advantage

        return returns, advantage

    def clear_memory(self):
        self.memory = Memory()

    def store(self, state, action, reward, mask):
        self.memory.store(self, state, action, reward, mask)

    def get_action(self, obs):
        return self.net.get_action(obs)

    def learn(self):
        states, actions, rewards, masks = self.memory.sample()

        policies, values = self.net(states)
        policies = policies.view(-1, self.num_actions)
        values = values.view(-1)

        returns, advantages = get_gae(values.view(-1).detach(), rewards, masks)

        log_policies = (torch.log(policies) * actions.detach()).sum(dim=1)
        actor_loss = -(log_policies * advantages).sum()
        critic_loss = (returns.detach() - values).pow(2).sum()
        
        entropy = (torch.log(policies) * policies).sum(1).sum()

        loss = actor_loss + self.ciritic_coefficient * critic_loss - self.entropy_coefficient * entropy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

def train(epochs=1000):
    env = gym.make('CartPole-v1')

    obs_dims = env.observation_space.shape[0]
    num_actions = env.action_space.n

    agent = A2C_Agent(num_actions, obs_dim)

    running_score = 0
    steps = 0
    loss = 0

    for e in range(epochs):
        done = False
        agent.clear_memory()

        score = 0
        state = env.reset()

        while not done:
            steps += 1

            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            mask = 0 if done else 1
            reward = reward if not done or score == 499 else -1

            agent.store(state, action, reward, mask)

            score += reward
            state = next_state

        loss = GAE.train_model(net, memory.sample(), optimizer)