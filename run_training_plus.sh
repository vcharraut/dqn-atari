#!/bin/bash

nohup sudo python3 learn.py --doubleq 1 --dueling 1 --adam 1 --mse 1 &
nohup sudo python3 learn.py --doubleq 1 --dueling 1 --adam 0 --mse 1 &
nohup sudo python3 learn.py --doubleq 1 --dueling 1 --adam 1 --mse 0 &
nohup sudo python3 learn.py --doubleq 1 --dueling 1 --adam 0 --mse 0 &
