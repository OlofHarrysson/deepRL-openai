import numpy as np

class Trainer():
  def __init__(self, env, agent, max_episode_length = 1000):
    self.env = env
    self.agent = agent
    self.max_episode_length = max_episode_length

  def run_episode(self, episode_numer, render):
    state = self.env.reset()
    state = np.reshape(state, [1, self.agent.get_state_size()])

    for t in range(self.max_episode_length):
      action = self.agent.act(state, training=True)
      next_state, reward, done = self.take_step(action)

      if render:
        self.env.render()

      self.agent.add_memory(state, action, reward, next_state, done)
      state = next_state

      if done:
        break

    self.agent.replay_memory()
    return t


  def take_step(self, action):
    next_state, reward, done, _ = self.env.step(action)
    reward = reward if not done else -20
    next_state = np.reshape(next_state, [1, self.agent.get_state_size()])
    return next_state, reward, done