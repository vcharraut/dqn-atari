import torch
import random

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

class ReplayMemory():

	def __init__(self, capacity):
		self.memory = []
		self.capacity = capacity


	def add(self, state, action, next_state, reward, end_step):
		# If the memory is full, we delete the first element
		if len(self.memory) > self.capacity:
			del self.memory[0]

		# Add the information to the memory
		self.memory.append((FloatTensor([state]),
							LongTensor([action]),
							FloatTensor([next_state]),
							FloatTensor([reward]),
							FloatTensor([end_step])))


	"""
	Return a random sample from self.memory of len = number
	"""
	def sample(self, number):
		sample = random.sample(self.memory, number)

		# Unwrap the batch to get the variables
		state, action, next_state, reward, done = zip(*sample)

		# Transform the variables to fit the computation
		state = torch.stack(state).squeeze()
		action = torch.stack(action)
		next_state = torch.stack(next_state).squeeze()
		reward = torch.stack(reward).squeeze()
		done = torch.stack(done).squeeze()

		return state, action, next_state, reward, done


	@property
	def size(self):
		return len(self.memory)
