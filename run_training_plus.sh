#!/bin/bash

sudo python3 learn_dqn.py --doubleq 1 --dueling 1 --adam 0 --mse 1
sudo python3 learn_dqn.py --doubleq 1 --dueling 1 --adam 1 --mse 1
sudo python3 learn_dqn.py --doubleq 1 --dueling 1 --adam 1 --mse 0
sudo python3 learn_dqn.py --doubleq 1 --dueling 1 --adam 0 --mse 0
