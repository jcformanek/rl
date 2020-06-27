import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import matplotlib.pyplot as plt

##  This code is adapted from @philtabor

class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), 
                                    dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), 
                                    dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, 
                                    dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, 
                                dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, 
                                dtype=np.int64)
        
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

class duelingDQN(nn.Module):
    def __init__(self, lr, n_actions, input_dims):
        super(duelingDQN, self).__init__()
        self.fc1 = nn.Linear(*input_dims, 32)
        self.V = nn.Linear(32, 1)
        self.A = nn.Linear(32, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()


    def forward(self, state):
        flat1 = F.relu(self.fc1(state))
        V = self.V(flat1)
        A = self.A(flat1)

        return V, A

class Agent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                    mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                    replace=1000):
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.learn_step_counter = 0
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace

        self.action_space = [i for i in  range(self.n_actions)]

        self.memory = ReplayBuffer(mem_size, input_dims)

        self.q_eval = duelingDQN(self.lr, self.n_actions, input_dims=self.input_dims)

        self.q_next = duelingDQN(self.lr, self.n_actions, input_dims=self.input_dims)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float)
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
    
    def decrement_epsilon(self):
        self.epsilon = self.epsilon -  self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states)
        actions = T.tensor(actions)
        dones = T.tensor(dones)
        rewards = T.tensor(rewards)
        states_ = T.tensor(states_)

        indices = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)

        V_s_eval, A_s_eval = self.q_eval.forward(states_)

        q_pred = T.add(V_s,
                        (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]

        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))
        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)

        q_target = rewards + (1 - dones)*self.gamma*q_next[indices, max_actions]

        loss = self.q_eval.loss(q_target, q_pred)

        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    no_games = 1000
    agent = Agent(gamma=0.99, epsilon=1.0, 
                    lr=5e-3, input_dims=[4], n_actions=2, 
                    mem_size=1000000, eps_min=0.01, 
                    batch_size=64, eps_dec=1e-4, 
                    replace=100)

    scores = []
    avg_scores = []
    for i in range(no_games):
        done = False
        observation = env.reset()
        score = 0

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_

        scores.append(score)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        print(i, avg_score, agent.epsilon)

        if i >= 300 and i%50 ==0: 
            plt.plot(np.arange(len(avg_scores)), avg_scores)
            plt.xlabel("No. of games played")
            plt.ylabel("Avg. returns")
            plt.show()
    