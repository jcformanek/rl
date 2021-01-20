import numpy as np
import torch

from replay_buffer import ReplayBuffer
from q_net import QNet



class DqnAgent():
    def __init__(self, obs_dims, act_dim, lr=1e-3, gamma=0.99, replay_buffer_size=10000,
     batch_size=64, epsilon_min=0.01, epsilon_dec=5e-5, target_update_frequency=64):
        self.buffer = ReplayBuffer(replay_buffer_size, obs_dims)
        self.batch_size = batch_size
        self.q_eval = QNet(obs_dims, act_dim)
        self.q_target = QNet(obs_dims, act_dim)
        self.obs_dims = obs_dims
        self.act_dim = act_dim
        self.learn_ctr = 0
        self.target_update_frequency = target_update_frequency
        self.gamma = gamma
        self.epsilon = 1
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.optimizer = torch.optim.Adam(self.q_eval.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss()


    def update_target(self):
        if self.learn_ctr % self.target_update_frequency == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())


    def decrement_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon - self.epsilon_dec


    def choose_action(self, obs):
        if np.random.sample() < self.epsilon:
            return np.random.randint(self.act_dim)
        else:
            obs = torch.tensor(np.expand_dims(obs, axis=0), dtype=torch.float)
            return torch.argmax(self.q_eval(obs)).item()


    def store_transition(self, obs, act, rew, _obs, done):
        self.buffer.push(obs, act, rew, _obs, done)


    def sample_replay_buffer(self):
        return self.buffer.sample(self.batch_size)


    def learn(self):
        self.optimizer.zero_grad()
        obs, act, rew, _obs, done = self.sample_replay_buffer()
        obs = torch.tensor(obs, dtype=torch.float)
        act = torch.tensor(act, dtype=torch.long)
        rew = torch.tensor(rew, dtype=torch.long)
        _obs = torch.tensor(_obs, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.long)
        idxs = torch.tensor(np.arange(self.batch_size), dtype=torch.long)
        q_pred = self.q_eval(obs)[idxs, act]
        q_next = self.q_target(_obs).max(dim=1)[0]
        q_target = rew + (1 - done) * self.gamma * q_next
        loss = self.loss_fn(q_target, q_pred)
        loss.backward()
        self.optimizer.step()
        self.update_target()
        self.decrement_epsilon()



dqn1 = DqnAgent((3, 21, 21), 4)
dqn2 = DqnAgent((3, 21, 21), 4)


def swap_axes(arr):
    arr = np.swapaxes(arr, 0, 2)
    arr = np.swapaxes(arr, 1, 2)
    return arr


from env_CatchPigs import EnvCatchPigs
import random

if __name__ == '__main__':
    env = EnvCatchPigs(7, False)
    max_iter = 1000000
    obs_list = env.get_obs()
    obs1 = swap_axes(obs_list[0])
    obs2 = swap_axes(obs_list[1])
    for i in range(max_iter):
        act1 = dqn1.choose_action(obs1)
        act2 = dqn2.choose_action(obs2)
        act_list = [act1, act2]
        # print("iter= ", i, env.agt1_pos, env.agt2_pos, env.pig_pos, env.agt1_ori, env.agt2_ori, 'action', act1, act2)
        # env.render()
        rew_list, done = env.step(act_list)
        rew1 = rew_list[0]
        rew2 = rew_list[1]
        # print(rew1)
        _obs_list = env.get_obs()
        _obs1 = swap_axes(_obs_list[0])
        _obs2 = swap_axes(_obs_list[1])
        dqn1.store_transition(obs1, act1, rew1, _obs1, done)
        dqn2.store_transition(obs2, act2, rew2, _obs2, done)
        obs1 = _obs1
        obs2 = _obs2
        dqn1.learn()
        dqn2.learn()
        #env.plot_scene()
        if rew_list[0] > 0:
            print("iter= ", i)
            print("agent 1 finds goal")

        if i % 1000 == 0:
            print("epsilon:", dqn1.epsilon)            