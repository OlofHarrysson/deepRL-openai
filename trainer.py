import numpy as np

class Trainer():
  def __init__(self, env, agent, episode_length):
    self.env = env
    self.agent = agent
    self.episode_length = episode_length


  def run_episode(self, episode_numer, render):
    state = self.env.reset()
    state = np.reshape(state, [1, self.agent.get_state_dim()])

    score = 0
    for t in range(self.episode_length):
      action = self.agent.act(state, training=True)
      next_state, reward, done = self.take_step(action[0])
      score += reward[0]

      self.agent.train(state, action, reward, next_state, done)
      state = next_state

      if render:
        self.env.render()

      if done:
        break

    return score


  def take_step(self, action):
    next_state, reward, done, _ = self.env.step(action)
    next_state = np.reshape(next_state, [1, self.agent.get_state_dim()])
    reward = np.reshape(reward, [1])
    return next_state, reward, done