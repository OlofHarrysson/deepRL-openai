#!/bin/bash
for i in {1..50}; do
  python main.py -n 2000 -p -s
  # python main.py -a ddpg -n 500 -e Pendulum-v0 -p -s
done