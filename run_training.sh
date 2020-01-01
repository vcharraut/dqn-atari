#!/bin/bash

nohup sudo python3 learn.py --doubleq 0 --dueling 0 --adam 1 --mse 1
nohup sudo python3 learn.py --doubleq 0 --dueling 0 --adam 0 --mse 1
nohup sudo python3 learn.py --doubleq 0 --dueling 0 --adam 1 --mse 0
nohup sudo python3 learn.py --doubleq 0 --dueling 0 --adam 0 --mse 0