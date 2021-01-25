import torch
import torch.distributions as D
import torch.nn as nn
import numpy as np
import gym
import matplotlib.pyplot as plt

def create_network(state_dims, action_dims):
    return nn.Sequential(nn.Linear(state_dims, 32),
                                    nn.ReLU(),
                                    nn.Linear(32, action_dims))

# policy = create_network(4,2)

# inp = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.float)

# logits = policy(inp)
# print(logits)
# dist = D.Categorical(logits=logits)
# print(dist.log_prob(torch.tensor([0,1])))
# logp = dist.log_prob(torch.tensor([0], dtype=torch.float32))


class REINFORCE_Agent():
    def __init__(self, state_dims, action_dims, lr=0.01):
        self.Logits_net = create_network(state_dims, action_dims)
        self.optimizer = torch.optim.Adam(self.Logits_net.parameters(), lr=lr)

    def get_policy(self, states):
        logits = self.Logits_net(states)
        return D.Categorical(logits=logits)

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action = self.get_policy(state).sample().item()
        return action

    def loss(self, states, actions, weights):
        log_probs = self.get_policy(states).log_prob(actions)
        return -(log_probs*weights).mean()

    def reinforce(self, states, actions, weights):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)
        self.optimizer.zero_grad()
        batch_loss = self.loss(states, actions, weights)
        batch_loss.backward()
        self.optimizer.step()
        return batch_loss

if __name__=="__main__":
    env = gym.make('CartPole-v1')
    agent = REINFORCE_Agent(4,2, lr=0.01)
    epochs = 100
    scores = []
    batch_size = 3000
    avg_scores = []
    for e in range(epochs):
        batch_states = []
        batch_actions = []
        batch_returns = []
        while True:
            done = False
            state = env.reset()
            score = 0
            rewards = []
            while not done:
                action = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                batch_states.append(state)
                batch_actions.append(action)
                rewards.append(reward)
                state = next_state
                score += 1
            cumulative_return = sum(rewards)
            for i in range(score):
                batch_returns.append(cumulative_return)
            scores.append(score)
            if len(batch_states) >= batch_size:
                break
        agent.reinforce(batch_states, batch_actions, batch_returns)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)

        print('Epoch: ', e,
             ' average score %.1f' % avg_score)

    plt.plot(np.arange(0,len(avg_scores)), avg_scores)
    plt.xlabel('No. of games played')
    plt.ylabel('Avg. returns')
    plt.show()
    print('done')
        



agent = REINFORCE_Agent(4, 2)
print(agent.reinforce([[1,2,3,4],[1,3,4,5]],[1,0], [10, 11]))