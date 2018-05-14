from agents.ddpg import DDPG_agent
from agents.dqn import DQN_agent
from agent_helpers.env_wrapper import pick_env
import random


def agent_creator(agent_type, env):
  env_info = pick_env(env)

  picker = {
    'ddpg': DDPG_agent,
    'dqn': DQN_agent
  }

  agent = picker.get(agent_type)

  if not agent:
    available_agents = list(picker.keys())
    raise Exception("{} is not a supported agent. Choose between {}".format(agent_type, ", ".join(available_agents)))

  learning_rate, learning_rate_decay, gamma, tau = generate_random_params()
  return agent(env_info, lr = learning_rate, gamma = gamma, tau = tau, lr_decay = learning_rate_decay)


def generate_random_params():
  a = random.uniform(1,10)
  b = random.randint(-5, -1)
  learning_rate = log_random(a, b)

  a = random.uniform(1,10)
  b = random.randint(2, 5)
  learning_rate_decay = log_random(a, b)

  a = random.uniform(1,10)
  b = random.randint(-4, -2)
  gamma = 1 - log_random(a, b)

  a = random.uniform(1,10)
  b = random.randint(-4, -1)
  tau = log_random(a, b)

  # print("lr {}".format(learning_rate))
  # print("lr_decay {}".format(learning_rate_decay))
  # print("gamma {}".format(gamma))
  # print("tau {}".format(tau))
  # asd

  return learning_rate, learning_rate_decay, gamma, tau

def log_random(a, b):
  return a * 10**(b)

