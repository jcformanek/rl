import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym

def train(env_name='CartPole-v1', hidden_dims=32, lr=0.01, batch_size=5000, epochs=100):
    env = gym.make(env_name)
    obs_dims = env.observation_space.shape[0]
    act_dims = env.action_space.n

    logits = nn.Sequential(nn.Linear(obs_dims, hidden_dims),
                            nn.ReLU(),
                            nn.Linear(hidden_dims, act_dims)
                            )

    value = nn.Sequential(nn.Linear(obs_dims, hidden_dims),
                            nn.ReLU(),
                            nn.Linear(hidden_dims, 1)
                            )

    logits_optimizer = optim.Adam(logits.parameters(), lr=lr)
    value_optimizer = optim.Adam(value.parameters(), lr=lr)

    def get_action(obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        policy = torch.distributions.Categorical(logits=logits(obs))
        return policy.sample().item()

    def get_value(obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        return value(obs).item()

    def get_rewards_to_go(rewards):
        n = len(rewards)
        rtgs = np.zeros_like(rewards)
        for i in reversed(range(n)):
            rtgs[i] = rewards[i] + (rtgs[i+1] if i+1 < n else 0)
        return list(rtgs)

    def get_value_loss(obs, rtgs):
        return ((value(obs) - rtgs)**2).mean()

    def get_logits_loss(obs, acts, weights):
        policy = torch.distributions.Categorical(logits=logits(obs))
        log_probs = policy.log_prob(acts)
        return -(log_probs*weights).mean()

    def train_one_epoch():
        batch_obs = []
        batch_acts = []
        batch_rtgs = []
        batch_vals = []
        scores = []
        while True:
            ep_rewards = []
            ep_len = 0
            obs = env.reset()
            done = False
            while not done:
                act = get_action(obs)
                val = get_value(obs)
                obs_, r, done, _ = env.step(act)

                batch_obs.append(obs)
                batch_acts.append(act)
                batch_vals.append(val)
                ep_rewards.append(r)

                obs = obs_
                ep_len += 1

            scores.append(ep_len)
            batch_rtgs += get_rewards_to_go(ep_rewards)

            if len(batch_obs) >= batch_size:
                break

        batch_obs = torch.tensor(batch_obs, dtype=torch.float32)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float32)
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float32)
        batch_vals = torch.tensor(batch_vals, dtype=torch.float32)

        logits_optimizer.zero_grad()
        batch_loss = get_logits_loss(batch_obs, batch_acts, (batch_rtgs-batch_vals))
        batch_loss.backward()
        logits_optimizer.step()

        value_optimizer.zero_grad()
        val_loss = get_value_loss(batch_obs, batch_rtgs)
        val_loss.backward()
        value_optimizer.step()

        return batch_loss, scores

    for e in range(epochs):
        loss, scores = train_one_epoch()
        avg_score = sum(scores) / len(scores)
        print('epoch: %3d \t loss: %.3f \t avg_score: %.3f'% (e, loss, avg_score))

if __name__ == '__main__':
    train(env_name='Acrobot-v1', lr=0.01)