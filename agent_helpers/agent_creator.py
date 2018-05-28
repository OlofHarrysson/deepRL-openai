from agents.ddpg import DDPG_agent
from agents.dqn import DQN_agent
from agent_helpers.env_wrapper import pick_env
import random


def agent_creator(agent_type, env, gen_random):
  env_info = pick_env(env)

  picker = {
    'ddpg': DDPG_agent,
    'dqn': DQN_agent
  }

  agent = picker.get(agent_type)

  if not agent:
    available_agents = list(picker.keys())
    raise Exception("{} is not a supported agent. Choose between {}".format(agent_type, ", ".join(available_agents)))

  if gen_random:
    if agent_type == 'dqn':
      learning_rate, learning_rate_decay, gamma, tau = DQN_parameters()
      return agent(env_info, lr = learning_rate, lr_decay = learning_rate_decay,
                   gamma = gamma, tau = tau)
    elif agent_type == 'ddpg':
      actor_p, critic_p, gamma = DDPG_parameters()
      return agent(env_info, actor_parameters = actor_p, critic_parameters = critic_p,
                   gamma = gamma)
    else:
      print("Agent doesn't exist, check agent_creator.py") # TODO: Do all this properly :)?

  return agent(env_info)


def DQN_parameters():
  a = random.uniform(1,10)
  b = random.randint(-5, -1)
  learning_rate = log_random(a, b)

  a = random.uniform(1,10)
  b = random.randint(2, 3)
  learning_rate_decay = log_random(a, b)

  a = random.uniform(1,10)
  b = random.randint(-4, -2)
  gamma = 1 - log_random(a, b)

  a = random.uniform(1,10)
  b = random.randint(-2, -1)
  tau = log_random(a, b)

  return learning_rate, learning_rate_decay, gamma, tau


def DDPG_parameters():
  # Generate actor parameters
  a = random.uniform(1,10)
  b = random.randint(-7, -1)
  lr = log_random(a, b)

  a = random.uniform(1,10)
  b = random.randint(2, 5)
  lr_decay = log_random(a, b)

  a = random.uniform(1,10)
  b = random.randint(-4, -1)
  tau = log_random(a, b)

  actor_p = {}
  actor_p['lr'] = lr
  actor_p['lr_decay'] = lr_decay
  actor_p['tau'] = tau

  # Generate critic parameters
  a = random.uniform(1,10)
  b = random.randint(-5, -1)
  lr = log_random(a, b)

  a = random.uniform(1,10)
  b = random.randint(2, 5)
  lr_decay = log_random(a, b)

  a = random.uniform(1,10)
  b = random.randint(-4, -1)
  tau = log_random(a, b)

  critic_p = {}
  critic_p['lr'] = lr
  critic_p['lr_decay'] = lr_decay
  critic_p['tau'] = tau

  # Generate other parameters
  a = random.uniform(1,10)
  b = random.randint(-4, -2)
  gamma = 1 - log_random(a, b)

  return actor_p, critic_p, gamma

def log_random(a, b):
  return a * 10**(b)

