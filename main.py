import argparse
import gym
from agent_helpers.agent_creator import agent_creator
from agent_helpers.trainer import Trainer

def parse():
  p = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

  p.add_argument("-n", "--n_train_episodes", type=int,
                 help="Number of episodes the agent is going to run")

  p.add_argument("-m", "--max_episode_length", type=int,
                 help="Number of steps in an episode")

  p.add_argument("-r", "--render_freq", type=int,
                 help="Render the environment every x-th episode")

  p.add_argument("-s", "--save", action="store_true",
                 help="Saves the model, score and parameters used")

  p.add_argument("-l", "--load_path", type=str,
                 help="Loads a previous model from path")

  p.add_argument("-a", "--agent_type", type=str,
                 help="Specifies which agent to use. Currently supports: ddpg, dqn")

  p.add_argument("-e", "--env_type", type=str,
                 help="Specifies which environment to use. Currently supports: CartPole-v1, Pendulum-v0, LunarLanderContinuous-v2")

  args = p.parse_args()
  return vars(args)


def main(n_train_episodes = 500, episode_length = 500, render_freq = 20,
         save = 'overwritable', load_path = None, record = False,
         agent_type = 'dqn', env_type = 'CartPole-v1'):

  # env = gym.make('CartPole-v1')
  # env = gym.make('LunarLander-v2')
  # env = gym.make('LunarLanderContinuous-v2')
  # env = gym.make('Pendulum-v0')

  env = gym.make(env_type) # TODO: Check the env_type input / give options
  agent = agent_creator(agent_type, env)

  # Record videos & more
  if record:
    env = gym.wrappers.Monitor(env, './saves/last_run', force=True)

  # Train agent
  trainer = Trainer(env, agent, n_train_episodes, episode_length, render_freq)

  n_test_episodes = 30
  if load_path: # TODO: loading-mode that doesn't update network
    agent.load(load_path) # Load pre-trained agent
    trainer.test(n_test_episodes)
  else:
    trainer.train(n_train_episodes)
    trainer.test(n_test_episodes)
    
    agent.save(save) # Save trained agent

  # Clean up
  if record:
    env.close()


if __name__ == "__main__":
  args = parse()
  main(**args)