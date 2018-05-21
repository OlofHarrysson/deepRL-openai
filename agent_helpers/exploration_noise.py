import numpy as np

class Ornstein_uhlenbeck_noise:
  def __init__(self, mu, nbr_episodes, sigma=0.3, theta=.15, dt=1e-2, x0=None):
    self.theta = theta
    self.mu = mu
    self.sigma = sigma
    self.dt = dt
    self.x0 = x0
    self.reset()

    # Stolen from e-greedy
    self.multiplier = 1.
    self.min_multiplier = 0.1
    fraction = 0.8
    self.decay_multi = self.min_multiplier ** (1 / (nbr_episodes * fraction))

  def __call__(self):
    # return self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
        # self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)

    x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
          self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
    self.x_prev = x
    return x * self.multiplier if self.multiplier > self.min_multiplier else x * self.min_multiplier


  def reset(self):
    self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

  def __repr__(self):
    return 'Ornstein_uhlenbeck_noise(mu={}, sigma={})'.format(self.mu, self.sigma)

  def reduce_noise(self):
    self.multiplier *= self.decay_multi

  def set_to_minimum(self):
    self.multiplier = self.min_multiplier


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

if __name__ == "__main__":
  import matplotlib.pyplot as plt
  n_epi = 2000
  noise = Ornstein_uhlenbeck_noise(np.zeros(1), n_epi, sigma=0.5, theta=0.9)

  ys = []
  xs = []
  for x in range(n_epi):
    xs.append(x)
    ys.append(noise())
    noise.reduce_noise()


  plt.plot(xs, ys)
  plt.show()
