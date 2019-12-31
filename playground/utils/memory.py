import torch
from random import sample
from collections import deque
import numpy as np

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

class ReplayMemory():

	def __init__(self, capacity):
		self.memory = deque(maxlen=capacity)


	def push(self, state, action, next_state, reward, done):
		self.memory.append((state, action, reward, next_state, done))


	"""
	Return a random sample from self.memory of len = number
	"""
	def sample(self, number):
		batch = sample(self.memory, number)

		# Unwrap the batch to get the variables
		state, action, next_state, reward, done = zip(*batch)

		return (np.array(state),
                np.array(action),
                np.array(reward, dtype=np.float32),
                np.array(next_state),
                np.array(done, dtype=np.uint8))


	@property
	def size(self):
		return len(self.memory)
