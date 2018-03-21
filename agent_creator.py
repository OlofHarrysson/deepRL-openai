from ddpg import DDPG_agent
from deep_q_agent import DQAgent
from env_wrapper import pick_env


def agent_creator(agent_string, env):
  env_info = pick_env(env)

  picker = {
    'ddpg': DDPG_agent,
    'deep_q': DQAgent
  }

  agent = picker.get(agent_string) # TODO: default value
  return agent(env_info)