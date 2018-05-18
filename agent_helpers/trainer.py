import numpy as np

class Trainer():
  def __init__(self, env, agent, nbr_episodes, episode_length, render_freq, logger):
    self.env = env
    self.agent = agent
    self.episode_length = episode_length
    self.render_freq = render_freq
    self.noise_generator = agent.create_noise_generator(nbr_episodes)
    self.logger = logger


  def train(self, n_episodes):
    self.fill_memory()
    print("Training ...")
    self._run_episodes(n_episodes, training=True)


  def test(self, n_episodes):
    self.noise_generator.set_to_minimum()
    print("Testing ...")
    total_score = self._run_episodes(n_episodes, training=False)
    return total_score


  def fill_memory(self):
    # Fills agents memory until it's big enough to train on
    while len(self.agent.memory) < self.agent.batch_size:
      self.random_agent()


  def _run_episodes(self, n_episodes, training):
    combined_score = 0
    for n_episode in range(1, n_episodes + 1):
      render = n_episode % self.render_freq == 0
      score = self._run_episode(n_episode, training, render)
      combined_score += score
      self.noise_generator.reduce_noise()

      print("Episode: {}/{}     Score: {:.2f}".format(n_episode,
        n_episodes, score))

    return combined_score


  def _run_episode(self, episode_number, training, render):
    state = self.env.reset()
    state = np.reshape(state, [1, self.agent.get_state_dim()])

    score = 0
    for t in range(self.episode_length):
      noise = self.noise_generator()
      action = self.agent.act(state, noise)
      next_state, reward, done = self.take_step(action[0])
      score += reward[0]

      if training:
        self.agent.add_memory(state, action, reward, next_state, done)

      state = next_state

      if render:
        self.env.render()

      if done:
        break

    if training:
      # TODO: range could be int(501.0 - score)? Works for all envs?
      for _ in range(10): # TODO, meta parameter?
        self.agent.train(self.logger)
        self.logger.add(episode_number, score, noise)
    else:
      self.logger.add_test(episode_number, score)
    
    return score


  def random_agent(self):
    state = self.env.reset()
    state = np.reshape(state, [1, self.agent.get_state_dim()])

    for t in range(self.episode_length):
      action = [self.env.action_space.sample()] # TODO: works for DDPG?
      next_state, reward, done = self.take_step(action[0])

      self.agent.add_memory(state, action, reward, next_state, done)
      state = next_state

      if done:
        break


  def take_step(self, action):
    next_state, reward, done, _ = self.env.step(action)
    next_state = np.reshape(next_state, [1, self.agent.get_state_dim()])
    reward = np.reshape(reward, [1])
    return next_state, reward, done