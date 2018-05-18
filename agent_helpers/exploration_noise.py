import numpy as np

class Ornstein_uhlenbeck_noise:
  def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
    self.theta = theta
    self.mu = mu
    self.sigma = sigma
    self.dt = dt
    self.x0 = x0
    self.reset()

  def __call__(self):
    return self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
        self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)

  def reduce_noise(self):
    self.x_prev = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
        self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)

  def reset(self):
    self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

  def __repr__(self):
    return 'Ornstein_uhlenbeck_noise(mu={}, sigma={})'.format(self.mu, self.sigma)

  def set_to_minimum(self):
    pass # TODO: How to implement this method? Random uniform noise?


class Epsilon_greedy:
  def __init__(self, nbr_episodes, e_min = 0.05, fraction = 0.8):
    # TODO: Incorporate start value for epsilon other than 1?
    self.epsilon_min = e_min
    self.epsilon = 1.
    self.epsilon_decay = e_min ** (1 / (nbr_episodes * fraction))

  def __call__(self):
    return self.epsilon if self.epsilon > self.epsilon_min else self.epsilon_min

  def reduce_noise(self):
    self.epsilon *= self.epsilon_decay

  def set_to_minimum(self):
    self.epsilon = self.epsilon_min

