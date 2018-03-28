import numpy as np

class Trainer():
  def __init__(self, env, agent, nbr_episodes, episode_length, render_freq):
    self.env = env
    self.agent = agent
    self.nbr_episodes = nbr_episodes
    self.episode_length = episode_length
    self.render_freq = render_freq
    self.noise_generator = agent.create_noise_generator(nbr_episodes)


  def train(self):
    for n_episode in range(1, self.nbr_episodes):
      render = n_episode % self.render_freq == 0
      score = self._run_episode(n_episode, render=render)
      self.noise_generator.reduce_noise()

      print("Episode: {}/{}     Score: {:.2f}".format(n_episode,
        self.nbr_episodes, score))


  def _run_episode(self, episode_numer, render):
    state = self.env.reset()
    state = np.reshape(state, [1, self.agent.get_state_dim()])

    score = 0
    for t in range(self.episode_length):
      noise = self.noise_generator()
      action = self.agent.act(state, noise, training=True)
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