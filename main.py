import argparse
import gym
from trainer import Trainer

def parse():
  p = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

  p.add_argument("-n", "--nbr_episodes", type=int,
                      help="Number of episodes the agent is going to run")

  p.add_argument("-e", "--episode_length", type=int,
                      help="Number of steps in an episode")

  p.add_argument("-r", "--render_freq", type=int,
                      help="Render the environment every x-th episode")

  p.add_argument("-s", "--save", action="store_true",
                      help="Saves the model, score and parameters used")

  p.add_argument("-l", "--load_path", type=str,
                      help="Loads a previous model from path")

  args = p.parse_args()
  return vars(args)

def main(nbr_episodes = 1000, episode_length = 500, render_freq = 20,
         save = 'overwritable', load_path = None):

  # from deep_q_agent import DQAgent
  # env = gym.make('CartPole-v1')
  # state_dim = env.observation_space.shape[0]
  # action_dim = env.action_space.n
  # agent = DQAgent(state_dim, action_dim)

  # env = gym.make('LunarLanderContinuous-v2')

  from ddpg import DDPG_agent
  env = gym.make('Pendulum-v0')
  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.shape[0]
  action_bound = env.action_space.high
  agent = DDPG_agent(state_dim, action_dim, action_bound)

  # env = gym.wrappers.Monitor(env, './saves', video_callable=lambda episode_id: episode_id%10==0) # TODO
  env = gym.wrappers.Monitor(env, './saves/last_run', force=True) # TODO

  if load_path:
    agent.load(load_path)

  trainer = Trainer(env, agent, episode_length)

  for n_episode in range(nbr_episodes):
    score = trainer.run_episode(n_episode, render=n_episode % render_freq == 0) # TODO: How to turn of rendering?
    print("Episode: {}/{}     Score: {:.2f}".format(n_episode, nbr_episodes, score))

  env.close()
  agent.save(save)


if __name__ == "__main__":
  args = parse()
  main(**args)