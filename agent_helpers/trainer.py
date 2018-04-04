import numpy as np
from agent_helpers.logger import Logger

class Trainer():
  def __init__(self, env, agent, nbr_episodes, episode_length, render_freq):
    self.env = env
    self.agent = agent
    self.nbr_episodes = nbr_episodes
    self.episode_length = episode_length
    self.render_freq = render_freq
    self.noise_generator = agent.create_noise_generator(nbr_episodes)
    self.logger = Logger()


  def train(self):
    for n_episode in range(1, self.nbr_episodes + 1):
      render = n_episode % self.render_freq == 0
      score = self._run_episode(n_episode, render=render)
      self.noise_generator.reduce_noise()

      print("Episode: {}/{}     Score: {:.2f}".format(n_episode,
        self.nbr_episodes, score))



  def _run_episode(self, episode_number, render):
    state = self.env.reset()
    state = np.reshape(state, [1, self.agent.get_state_dim()])



    losses = []
    max_qs = np.array([])


    score = 0
    for t in range(self.episode_length):
      noise = self.noise_generator()
      action = self.agent.act(state, noise, training=True)
      next_state, reward, done = self.take_step(action[0])
      score += reward[0]

      loss, max_q = self.agent.train(state, action, reward, next_state, done)
      losses.append(loss)
      max_qs = np.append(max_qs, max_q)
      
      state = next_state

      if render:
        self.env.render()

      if done:
        break


    if episode_number > 10: # TODO, 

      log_data = {}
      log_data['noise'] = self.noise_generator()
      log_data['score'] = score

      max_loss = max(losses) # TODO numpy, faster
      min_loss = min(losses)
      avg_loss = sum(losses) / len(losses)

      log_data['max_loss'] = max_loss
      log_data['min_loss'] = min_loss
      log_data['avg_loss'] = avg_loss

      max_q = np.amax(max_qs) # TODO numpy, faster
      min_q = np.min(max_qs)
      avg_q = np.sum(max_qs) / len(max_qs)

      log_data['max_q'] = max_q
      log_data['min_q'] = min_q
      log_data['avg_q'] = avg_q

      # self.logger.add(log_data, episode_number)
      self.logger.add2(log_data, episode_number)


    return score


  def take_step(self, action):
    next_state, reward, done, _ = self.env.step(action)
    next_state = np.reshape(next_state, [1, self.agent.get_state_dim()])
    reward = np.reshape(reward, [1])
    return next_state, reward, done