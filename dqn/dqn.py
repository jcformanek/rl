import numpy as np
import random
import torch
import torch.nn as nn
import gym
import matplotlib.pyplot as plt

class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = []
        self.ptr = 0


    def __str__(self):
        return ("Size: " + str(self.size) + "\n"
                "Pointer: " + str(self.ptr) + "\n"
                "Buffer: " + str(self.buffer) + "\n")


    def insert(self, state, action, next_state, reward, done):
        if len(self.buffer) < self.size:
            self.buffer.append((state, action, next_state, reward, done))
        else:
            self.buffer[self.ptr] = (state, action, next_state, reward, done)
        self.ptr = (self.ptr + 1) % self.size


    def sample(self, batch_size):
        if len(self.buffer) >= batch_size:
            transitions = random.sample(self.buffer, batch_size)
        else:
            transitions = self.buffer
        states, actions, next_states, rewards, dones = [], [], [], [], []
        for T in transitions:
            s, a, n_s, r, d = T
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            next_states.append(np.array(n_s, copy=False))
            rewards.append(r)
            dones.append(d)
        return (np.array(states), np.array(actions), np.array(next_states),
                    np.array(rewards), np.array(dones))


def create_network(state_dims, action_dims):
    return nn.Sequential(nn.Linear(state_dims, 32),
                                    nn.ReLU(),
                                    nn.Linear(32, action_dims))

class DQN_Agent():
    def __init__(self, state_dims, action_dims, lr=1e-3, gamma=0.99, memory_size=1000000,
     batch_size=64, epsilon=1, epsilon_min=0.01, epsilon_dec=5e-5, target_update_frequency=64):
        self.memory = ReplayBuffer(memory_size)
        self.batch_size = batch_size
        self.Q_eval = create_network(state_dims, action_dims)
        self.Q_target = create_network(state_dims, action_dims)
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.learn_counter = 0
        self.target_update_frequency = target_update_frequency
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.optimizer = torch.optim.RMSprop(self.Q_eval.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss()


    def update_target(self):
        if self.learn_counter % self.target_update_frequency == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())


    def decrement_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon - self.epsilon_dec


    def choose_action(self, state):
        if np.random.sample() < self.epsilon:
            return np.random.randint(self.action_dims)
        else:
            state = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float)
            return torch.argmax(self.Q_eval(state)).item()


    def store_transition(self, state, action, next_state, reward, done):
        self.memory.insert(state, action, next_state, reward, done)


    def sample_memory_batch(self):
        states, actions, next_states, rewards, dones = self.memory.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions)
        next_states = torch.tensor(next_states, dtype=torch.float)
        rewards = torch.tensor(rewards, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.float)
        return (states, actions, next_states, rewards, dones)


    def learn(self):
        self.optimizer.zero_grad()
        states, actions, next_states, rewards, dones = self.sample_memory_batch()
        indexs = np.arange(len(states))
        q_pred = self.Q_eval(states)[indexs, actions]
        q_next = self.Q_target(next_states).max(dim=1)[0]
        q_target = rewards + (1 - dones) * self.gamma * q_next
        loss = self.loss_fn(q_target, q_pred)
        loss.backward()
        self.optimizer.step()
        self.update_target()
        self.decrement_epsilon()



if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = DQN_Agent(4, 2)
    epochs = 1000
    scores = []
    avg_scores = []
    for e in range(epochs):
        done = False
        state = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, next_state, reward, done)
            agent.learn()
            state = next_state
            score += reward
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)

        print('epoch: ', e, ' average score %.1f' % avg_score,
            'epsilon %.2f' % agent.epsilon)

    plt.plot(np.arange(0,len(avg_scores)), avg_scores)
    plt.xlabel('No. of games played')
    plt.ylabel('Avg. returns')
    plt.show()
    print('done')