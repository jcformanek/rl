from pettingzoo.mpe import simple_v2
import deep_q_learning 
import numpy as np

env = simple_v2.env()

env.reset()

model = deep_q_learning.Agent(4, 5, hidden_dims=[100])


def train(agent, env, num_games=100  00):
    ep_rets = []
    avg_ep_rets = []
    for i in range(num_games):
        ret = 0
        env.reset()
        obs = np.array([0.0,0.0,0.0,0.0])
        for agent in env.agent_iter():
            next_obs, rew, done, _ = env.last()
            if done:
                break
            act = model.choose_action(obs)
            env.step(act)
            model.store_transition(obs, act, rew, next_obs, done)
            obs = next_obs
            model.learn()
            ret += rew
            
            

        ep_rets.append(ret)
        avg_ep_rets.append(np.mean(ep_rets[-100:]))

        if (i+1) % 100 == 0:
            print(f"Game num: {i+1}    Avg score: {round(avg_ep_rets[-1], 2)}")

    return avg_ep_rets

train(model, env)