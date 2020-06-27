import numpy as np
import gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

MAXSTATES = 10**4
GAMMA = 0.9
ALPHA = 0.01

def max_dict(d):
    max_val = float('-inf')
    for key, val in d.items():
        if val > max_val:
            max_val = val
            max_key = key
    return max_key, max_val

def create_bins():
    bins = np.zeros((4,10))
    bins[0] = np.linspace(-4.8, 4.8, 10)
    bins[1] = np.linspace(-5, 5, 10)
    bins[2] = np.linspace(-0.418, 0.418, 10)
    bins[3] = np.linspace(-5, 5, 10)

    return bins

def assign_bins(obs, bins):
    state = np.zeros(4)
    for i in range(4):
        state[i] = np.digitize(obs[i], bins[i])
    return state

def get_state_as_string(state):
    string_state = ''.join(str(int(e)) for e in state)
    return string_state

def get_all_states_as_strings():
    states = []
    for i in range(MAXSTATES):
        states.append(str(i).zfill(4))
    return states

def initialize_Q():
    Q = {}
    all_states = get_all_states_as_strings()
    for state in all_states:
        Q[state] = {}
        for action in range(env.action_space.n):
            Q[state][action] = 0
    return Q

def play_one_game(bins, Q, eps=0.5):
    obs =env.reset()
    done = False
    cnt = 0
    state = get_state_as_string(assign_bins(obs, bins))
    total_reward = 0

    while not done:
        cnt += 1
        if np.random.uniform() < eps:
            act = env.action_space.sample()
        else:
            act = max_dict(Q[state])[0]
        
        obs, reward, done, _ = env.step(act)
        total_reward += reward

        if done and cnt < 200:
            reward = -300

        state_new = get_state_as_string(assign_bins(obs, bins))

        a1, max_q = max_dict(Q[state_new])

        Q[state][act] += ALPHA*(reward+GAMMA*max_q - Q[state][act])
        state, act = state_new, a1
    return total_reward, cnt

def play_many_games(bins, N=3000):
    Q =initialize_Q()
    length = []
    reward = []
    avg_rewards = []
    for n in range(N):
        eps = 1.0/np.sqrt(n+1)
        ep_reward, ep_len = play_one_game(bins, Q, eps)
        avg = sum(reward[-100:])/100
        avg_rewards.append(avg)
        print(n, avg)
        length.append(ep_len)
        reward.append(ep_reward)
    return length, avg_rewards

######### MAIN ###########
bins = create_bins()
lengths, avg_rewards = play_many_games(bins)

plt.plot(np.arange(len(avg_rewards)), avg_rewards)
plt.xlabel("No. of games played")
plt.ylabel("Avg. returns")
plt.show()