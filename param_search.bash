#!/bin/bash
for i in {1..10}; do
  # Examples of commands that can be ran. See main.py for input information
  # python main.py -a dqn -n 2000 -e LunarLander-v2 -m 750 -p -s
  # python main.py -a ddpg -n 5000 -e Pendulum-v0 -m 200 -p -s 
  
  python main.py -n 5000 -p -s
done