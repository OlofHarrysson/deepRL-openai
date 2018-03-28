from agents.ddpg import DDPG_agent
from agents.dqn import DQN_agent
from agent_helpers.env_wrapper import pick_env


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

  return agent(env_info)