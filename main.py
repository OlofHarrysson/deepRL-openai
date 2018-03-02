import argparse
import gym
from trainer import Trainer

def parse():
  parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

  parser.add_argument("-n", "--nbr_episodes", type=int,
                      help="Number of episodes the agent is going to run")

  parser.add_argument("-e", "--episode_length", type=int,
                      help="Number of steps in an episode")

  parser.add_argument("-r", "--render_freq", type=int,
                      help="Render the environment every x-th episode")

  parser.add_argument("-s", "--save", action="store_true",
                      help="Saves the model, score and parameters used")

  parser.add_argument("-l", "--load_path", type=str,
                      help="Loads a previous model from path")

  args = parser.parse_args()
  return vars(args)

def main(nbr_episodes = 1000, episode_length = 500, render_freq = 20,
         save = 'overwritable', load_path = None):

  from deep_q_agent import DQAgent

  env = gym.make('CartPole-v1')
  state_size = env.observation_space.shape[0]
  action_size = env.action_space.n
  agent = DQAgent(state_size, action_size)
  if load_path:
    agent.load(load_path)

  trainer = Trainer(env, agent)

  for n_episode in range(nbr_episodes):
    score = trainer.run_episode(n_episode, render=n_episode % render_freq == 0) # TODO: How to turn of rendering?
    print("Episode: {}/{}, score: {}".format(n_episode, nbr_episodes, score))

  agent.save(save)


if __name__ == "__main__":
  args = parse()
  main(**args)