import gym
import matplotlib.pyplot as plt

plt.plot([1,2,3,4,5,6,7])
plt.show()

# env = gym.make("CartPole-v1")
# observation = env.reset()
# for _ in range(1000):
#   env.render()
#   action = env.action_space.sample() # your agent here (this takes random actions)
#   observation, reward, done, info = env.step(action)

#   if done:
#     observation = env.reset()
# env.close()