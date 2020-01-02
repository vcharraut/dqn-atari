#!/bin/bash

sudo python3 learn_rainbow.py --minimal 1 --adam 1 
sudo python3 learn_rainbow.py --minimal 1 --adam 0 
sudo python3 learn_rainbow.py --minimal 0 --adam 1 
sudo python3 learn_rainbow.py --minimal 0 --adam 0
