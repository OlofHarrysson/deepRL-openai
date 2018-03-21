import argparse
import gym
from agent_creator import agent_creator
from trainer import Trainer

def parse():
  p = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

  p.add_argument("-n", "--nbr_episodes", type=int,
                 help="Number of episodes the agent is going to run")

  p.add_argument("-m", "--max_episode_length", type=int,
                 help="Number of steps in an episode")

  p.add_argument("-r", "--render_freq", type=int,
                 help="Render the environment every x-th episode")

  p.add_argument("-s", "--save", action="store_true",
                 help="Saves the model, score and parameters used")

  p.add_argument("-l", "--load_path", type=str,
                 help="Loads a previous model from path")

  p.add_argument("-a", "--agent", type=str,
                 help="Specifies which agent to use. Currently supports: ddpg, deep_q")

  p.add_argument("-e", "--env", type=str,
                 help="Specifies which environment to use. Currently supports: CartPole-v1, Pendulum-v0, LunarLanderContinuous-v2")

  args = p.parse_args()
  return vars(args)


def main(nbr_episodes = 1000, episode_length = 500, render_freq = 20,
         save = 'overwritable', load_path = None, record = False):

  # env = gym.make('CartPole-v1')
  # agent = agent_creator('deep_q', env)

  env = gym.make('LunarLanderContinuous-v2')
  # env = gym.make('Pendulum-v0')

  agent = agent_creator('ddpg', env)

  if record:
    env = gym.wrappers.Monitor(env, './saves/last_run', force=True)

  if load_path:
    agent.load(load_path)

  trainer = Trainer(env, agent, episode_length)

  for n_episode in range(nbr_episodes):
    score = trainer.run_episode(n_episode, render=n_episode % render_freq == 0) # TODO: How to turn of rendering?
    print("Episode: {}/{}     Score: {:.2f}".format(n_episode, nbr_episodes, score))

  if record:
    env.close()

  agent.save(save)


if __name__ == "__main__":
  args = parse()
  main(**args)