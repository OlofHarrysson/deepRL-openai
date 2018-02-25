import numpy as np
import sys
from deep_q_agent import DQAgent
import gym

if __name__ == "__main__":
  env = gym.make('CartPole-v1')
  state_size = env.observation_space.shape[0]
  action_size = env.action_space.n
  agent = DQAgent(state_size, action_size)
  agent.load("./save/cartpole-dqn.h5")

  n_episodes = 10
  for n_episode in range(n_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])

    for t in range(500):
      env.render()

      action = agent.act(state)

      state, reward, done, _ = env.step(action)
      state = np.reshape(state, [1, state_size])

      if done:
        print("Episode: {}/{}, score: {}".format(n_episode, n_episodes, t))
        break