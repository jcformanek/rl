import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym

def train(env_name='CartPole-v1', hidden_dims=32, lr=0.01, batch_size=5000, epochs=100, gamma=0.99):
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
        return value(obs)

    def get_rewards_to_go(rewards):
        n = len(rewards)
        rtgs = np.zeros_like(rewards)
        for i in reversed(range(n)):
            rtgs[i] = rewards[i] + gamma * (rtgs[i+1] if i+1 < n else 0)
        return list(rtgs)

    def get_value_loss(advs):
        advs = torch.stack(advs) 
        return advs.pow(2).mean()

    def get_logits_loss(obs, acts, advs):
        obs = torch.tensor(obs, dtype=torch.float32)
        acts = torch.tensor(acts, dtype=torch.float32)
        advs = torch.tensor(advs, dtype=torch.float32)
        policy = torch.distributions.Categorical(logits=logits(obs))
        log_probs = policy.log_prob(acts)
        return -(log_probs*advs).mean()

    def train_one_epoch():
        batch_obs = []
        batch_acts = []
        batch_advs = []
        batch_rtgs = []
        batch_ret = []
        batch_ep_lens = []
        while True:
            ep_rew = []
            ep_len = 0
            obs = env.reset()
            done = False
            while not done:
                act = get_action(obs)
                obs_, r, done, _ = env.step(act)
                adv = r + (1 - done) * gamma * get_value(obs_) - get_value(obs)

                batch_obs.append(obs)
                batch_acts.append(act)
                batch_advs.append(adv)
                ep_rew.append(r)

                obs = obs_
                ep_len += 1
            batch_ep_lens.append(ep_len)
            batch_ret.append(sum(ep_rew))
            batch_rtgs += get_rewards_to_go(ep_rew)

            if len(batch_obs) >= batch_size:
                break

        logits_optimizer.zero_grad()
        logits_loss = get_logits_loss(batch_obs, batch_acts, batch_advs)
        logits_loss.backward()
        logits_optimizer.step()

        value_optimizer.zero_grad()
        value_loss = get_value_loss(batch_advs)
        value_loss.backward()
        value_optimizer.step()

        return logits_loss, value_loss, batch_ret, batch_ep_lens

    for e in range(epochs):
        l_loss, v_loss, rets, ep_lens = train_one_epoch()
        avg_ep_len = sum(ep_lens) / len(ep_lens)
        avg_rets = sum(rets) / len(rets)
        print('epoch: %3d | Pi loss: %.3f | V loss: %.3f | avg return: %.3f | avg ep len: %.3f'%
                     (e, l_loss, v_loss, avg_rets, avg_ep_len))

if __name__ == '__main__':
    train(lr=0.1)