import gym
import numpy as np
import sys
from deep_q_agent import DQAgent
import argparse


def parse():
  parser = argparse.ArgumentParser()

  parser.add_argument('-m', action='store', dest='training',
                    help='Runs the program in training mode.')

  return parser


def take_step(env, action):
  # TODO: Position of stick instead of position of box?
  asdasd
  next_state, reward, done, _ = env.step(action)
  reward -= np.abs(next_state[0])
  reward = reward if not done else -10
  next_state = np.reshape(next_state, [1, state_size])

  np.argmax



  return next_state, reward, done


if __name__ == "__main__":
  env = gym.make('CartPole-v1')
  state_size = env.observation_space.shape[0]
  action_size = env.action_space.n
  agent = DQAgent(state_size, action_size)
  batch_size = 32

  n_episodes = 5000
  for n_episode in range(n_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])

    for t in range(500):
      action = agent.act(state, training=True)
      next_state, reward, done = take_step(env, action)
      

      agent.add_memory(state, action, reward, next_state, done)
      state = next_state
      if done:
        print("Episode: {}/{}, score: {}".format(n_episode, n_episodes, t))
        break


    if len(agent.memory) > batch_size:
      agent.replay_memory(batch_size)

  agent.save("./save/cartpole-dqn.h5")


