import torch
from random import sample
import numpy as np

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

class ReplayMemory():

	def __init__(self, capacity):
		self.memory = []
		self.capacity = capacity
		self.index = -1


	def push(self, action, next_state, reward, done):
		if len(self.memory) > self.capacity:
			del self.memory[0]

		self.index += 1
		self.memory.append((action, next_state, reward, done, self.index))


	"""
	Return a random sample from self.memory of len = number
	"""
	def sample(self, number):
		batch = sample(self.memory[1:], number)

		# Unwrap the batch to get the variables
		action, next_state, reward, done, index = zip(*batch)

		state = [self.memory[i-1][1] for i in index] 

		return (np.array(state),
                np.array(action),
                np.array(reward, dtype=np.float32),
                np.array(next_state),
                np.array(done, dtype=np.uint8))


	@property
	def size(self):
		return len(self.memory)