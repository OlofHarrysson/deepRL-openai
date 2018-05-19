#!/bin/bash
for i in {1..10}; do
  # python main.py -n 5000 -s -p
  python main.py -a ddpg -n 500 -e Pendulum-v0 -p -s
done