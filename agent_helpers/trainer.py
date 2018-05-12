import numpy as np
from agent_helpers.logger import Logger

class Trainer():
  def __init__(self, env, agent, nbr_episodes, episode_length, render_freq):
    self.env = env
    self.agent = agent
    self.episode_length = episode_length
    self.render_freq = render_freq
    self.noise_generator = agent.create_noise_generator(nbr_episodes)
    self.logger = Logger()


  def train(self, n_episodes):
    self.fill_memory()
    print("Training ...")
    self._run_episodes(n_episodes)

  def test(self, n_episodes):
    self.noise_generator.set_to_minimum()
    print("Testing ...")
    self._run_episodes(n_episodes)

  def fill_memory(self):
    # Fills agents memory until it's big enough to train on
    while len(self.agent.memory) < self.agent.batch_size:
      self.random_agent()


  def _run_episodes(self, n_episodes):
    for n_episode in range(1, n_episodes + 1):
      render = n_episode % self.render_freq == 0
      score = self._run_episode(n_episode, render=render)
      self.noise_generator.reduce_noise()

      print("Episode: {}/{}     Score: {:.2f}".format(n_episode,
        n_episodes, score))

  def _run_episode(self, episode_number, render, training):
    state = self.env.reset()
    state = np.reshape(state, [1, self.agent.get_state_dim()])

    score = 0
    for t in range(self.episode_length):
      noise = self.noise_generator()
      action = self.agent.act(state, noise, training=True)
      next_state, reward, done = self.take_step(action[0])
      score += reward[0]

      if training:
        self.agent.add_memory(state, action, reward, next_state, done)

      if render:
        self.env.render()

      state = next_state

      if done:
        break

    if training:
      losses, max_q = self._update_agent()
      self.logger.add(episode_number, score, noise, max_qs, losses)
    else:
      pass
      # Add testing logs

    
    return score

  def _update_agent(self):
    losses = []
    max_qs = np.array([])

    for _ in range(10): # TODO, meta parameter?
      loss, max_q = self.agent.train()
      losses.append(loss)
      max_qs = np.append(max_qs, max_q)

    return losses, max_qs


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


  # def _run_episode(self, episode_number, render):
  #   state = self.env.reset()
  #   state = np.reshape(state, [1, self.agent.get_state_dim()])

  #   losses = []
  #   max_qs = np.array([])
  #   actions = []

  #   score = 0
  #   for t in range(self.episode_length):
  #     noise = self.noise_generator()
  #     action = self.agent.act(state, noise, training=True)
  #     next_state, reward, done = self.take_step(action[0])
  #     score += reward[0]

  #     self.agent.add_memory(state, action, reward, next_state, done)
  #     loss, max_q = self.agent.train()      
  #     state = next_state

  #     if render:
  #       self.env.render()

  #     losses.append(loss)
  #     max_qs = np.append(max_qs, max_q)
  #     actions.append(action[0])

  #     if done:
  #       break

  #   self.logger.add(episode_number, score, noise, max_qs, losses, actions)
  #   return score


  def take_step(self, action):
    next_state, reward, done, _ = self.env.step(action)
    next_state = np.reshape(next_state, [1, self.agent.get_state_dim()])
    reward = np.reshape(reward, [1])
    return next_state, reward, done