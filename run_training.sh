#!/bin/bash

sudo python3 learn.py --doubleq 0 --dueling 0 --adam 1 --mse 1
sudo python3 learn.py --doubleq 0 --dueling 0 --adam 0 --mse 1
sudo python3 learn.py --doubleq 0 --dueling 0 --adam 1 --mse 0
sudo python3 learn.py --doubleq 0 --dueling 0 --adam 0 --mse 0