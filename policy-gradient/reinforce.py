import torch
import torch.distributions as D
import torch.nn as nn
import numpy as np
import gym

def create_network(state_dims, action_dims):
    return nn.Sequential(nn.Linear(state_dims, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, action_dims))

# policy = create_network(4,2)

# inp = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.float)

# logits = policy(inp)
# print(logits)
# dist = D.Categorical(logits=logits)
# print(dist.log_prob(torch.tensor([0,1])))
# logp = dist.log_prob(torch.tensor([0], dtype=torch.float32))


class REINFORCE_Agent():
    def __init__(self, state_dims, action_dims, lr=0.0001):
        self.Logits_net = create_network(state_dims, action_dims)
        self.optimizer = torch.optim.RMSprop(self.Logits_net.parameters(), lr=lr)

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
    agent = REINFORCE_Agent(4,2, lr=0.00001)
    epochs = 100000
    scores = []
    for e in range(epochs):
        states = []
        actions = []
        rewards = []
        done = False
        state = env.reset()
        score = 0
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            score += 1
        # cumulative_sum = np.zeros(len(rewards)) + sum(rewards)
        n = len(rewards)
        rtgs = np.zeros_like(rewards)
        for i in reversed(range(n)):
            rtgs[i] = rewards[i] + (rtgs[i+1] if i+1 < n else 0)
        rtgs = list(rtgs)
        agent.reinforce(states, actions, rtgs)
        scores.append(score)
        avg_score = np.mean(scores[-100:])

        print('episode: ', e,'score: ', score,
             ' average score %.1f' % avg_score)
        



agent = REINFORCE_Agent(4, 2)
print(agent.reinforce([[1,2,3,4],[1,3,4,5]],[1,0], [10, 11]))