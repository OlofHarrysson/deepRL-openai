import numpy as np

def pick_env(env):
  picker = {
    'LunarLanderContinuous': LunarLanderContinuous,
    'PendulumEnv': Pendulum,
    'CartPoleEnv': Cartpole,
    'LunarLander': LunarLander
  }

  env_helper = picker.get(env.env.__class__.__name__)

  if not env_helper:
    available_env = list(picker.keys())
    raise Exception("{} is not a supported environment. Choose between {}".format(env, ", ".join(available_env)))

  return env_helper(env)


# ~~~~~~~~~~~~~~  Superclass ~~~~~~~~~~~~~~
class Env_helper():
  def __init__(self, env):
    self.env = env

  def get_state_dim(self):
    return self.state_dim

  def get_action_dim(self):
    return self.action_dim

  def custom_reward(self, reward):
    ''' Customized reward function in subclasses'''
    return reward


# ~~~~~~~~~~~~~~ Env types ~~~~~~~~~~~~~~
class Continious_actionspace(Env_helper):
  def __init__(self, env):
    super().__init__(env)


  def get_action_bound(self):
    # TODO: Change if new inv is not centered around 0 or different actions have different ranges. Some agents probably also need to change
    low = self.env.action_space.low
    high = self.env.action_space.high
    assert np.array_equal(-low, high)
    return high


class Discrete_actionspace(Env_helper):
  def __init__(self, env):
    super().__init__(env)


# ~~~~~~~~~~~~~~ Environments ~~~~~~~~~~~~~~
class LunarLanderContinuous(Continious_actionspace):
  def __init__(self, env):
    super().__init__(env)
    self.state_dim = env.observation_space.shape[0]
    self.action_dim = env.action_space.shape[0]


class Pendulum(Continious_actionspace):
  def __init__(self, env):
    super().__init__(env)
    self.state_dim = env.observation_space.shape[0]
    self.action_dim = env.action_space.shape[0]


class Cartpole(Discrete_actionspace):
  def __init__(self, env):
    super().__init__(env)
    self.state_dim = env.observation_space.shape[0]
    self.action_dim = env.action_space.n

class LunarLander(Discrete_actionspace):
  def __init__(self, env):
    super().__init__(env)
    self.state_dim = env.observation_space.shape[0]
    self.action_dim = env.action_space.n

