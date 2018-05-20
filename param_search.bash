#!/bin/bash
for i in {1..20}; do
  python main.py -a dqn -n 2000 -e LunarLander-v2 -m 250 -p -s
  # python main.py -a ddpg -n 500 -e Pendulum-v0 -p -s
done