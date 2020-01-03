import torch
from random import sample
from collections import deque, namedtuple
import numpy as np


class ReplayMemory():

	def __init__(self, capacity):
		self.memory = deque(maxlen=capacity)

	"""Add an experience to the memory"""
	def push(self, state, action, next_state, reward, done):
		self.memory.append((state, action, next_state, reward, done))


	"""Return a random sample from self.memory of size = number"""
	def sample(self, number):
		batch = sample(self.memory, number)

		# Unwrap the batch to get the variables
		state, action, next_state, reward, done = zip(*batch)

		return (np.array(state),
				np.array(action),
				np.array(next_state),
				np.array(reward, dtype=np.float32),
				np.array(done, dtype=np.uint8))

	@property
	def size(self):
		return len(self.memory)


# Segment tree data structure where parent node values are sum/max of children node values
# REF : https://github.com/Kaixhin/Rainbow/blob/master/memory.py
class SegmentTree():

	def __init__(self, capacity):
		self.index = 0

		self.capacity = capacity

		# Used to track actual capacity
		self.full = False  

		# Initialise fixed size tree with all (priority) zeros
		self.sum_tree = np.zeros((2 * capacity - 1, ), dtype=np.float32)

		# Wrap-around cyclic buffer
		self.data = np.array([None] * capacity)  

		# Initial max value to return (1 = 1^ω)
		self.max = 1  


	"""Propagates value up tree given a tree index"""
	def _propagate(self, index, value):
		parent = (index - 1) // 2
		left, right = 2 * parent + 1, 2 * parent + 2
		self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
		if parent != 0:
			self._propagate(parent, value)


	"""Updates value given a tree index"""
	def update(self, index, value):
		# Set new value
		self.sum_tree[index] = value  

		# Propagate value
		self._propagate(index, value)  
		self.max = max(value, self.max)


	def append(self, data, value):
		# Store data in underlying data structure
		self.data[self.index] = data  

		# Update tree
		self.update(self.index + self.capacity - 1, value)  

		# Update index
		self.index = (self.index + 1) % self.capacity

		# Save when capacity reached
		self.full = self.full or self.index == 0  
		self.max = max(value, self.max)


	"""Searches for the location of a value in sum tree"""
	def _retrieve(self, index, value):
		left, right = 2 * index + 1, 2 * index + 2
		if left >= len(self.sum_tree):
			  return index
		elif value <= self.sum_tree[left]:
			  return self._retrieve(left, value)
		else:
			  return self._retrieve(right, value - self.sum_tree[left])


	"""Searches for a value in sum tree and returns value, data index and tree index"""
	def find(self, value):
		index = self._retrieve(0, value)  # Search for index of item from root
		data_index = index - self.capacity + 1
		# Return value, data index, tree index
		return (self.sum_tree[index], data_index, index)


	"""Returns data given a data index"""
	def get(self, data_index):
		return self.data[data_index % self.capacity]


	def total(self):
		return self.sum_tree[0]


Transition = namedtuple('Transition', ('timestep', 'state', 'action', 'reward', 'nonterminal'))
blank_trans = Transition(0, torch.zeros(84, 84, dtype=torch.uint8), None, 0, False)


# REF : https://github.com/Kaixhin/Rainbow/blob/master/memory.py
class PrioritizedReplayMemory():

	def __init__(self, capacity):
		self.device = torch.device('cuda')

		self.capacity = capacity

		self.history = 4

		self.discount = 0.99

		self.n = 3

		# Initial importance sampling weight β, annealed to 1 over course of training
		self.priority_weight = 0.4
		self.priority_exponent = 0.5

		# Internal episode timestep counter
		self.t = 0  
		
		# Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities
		self.transitions = SegmentTree(capacity)  


	"""Adds state and action at time t, reward and terminal at time t + 1"""
	def append(self, state, action, reward, terminal):
		# Only store last frame and discretise to save memory
		state = torch.from_numpy(state)
		state = state[-1].mul(255).to(dtype=torch.uint8, device=torch.device('cpu'))  

		# Store new transition with maximum priority
		self.transitions.append(Transition(self.t, state, action, reward, not terminal), self.transitions.max)  

		# Start new episodes with t = 0
		self.t = 0 if terminal else self.t + 1  


	"""Returns a transition with blank states where appropriate"""
	def _get_transition(self, idx):
		transition = np.array([None] * (self.history + self.n))
		transition[self.history - 1] = self.transitions.get(idx)

		for t in range(self.history - 2, -1, -1):  # e.g. 2 1 0
			if transition[t + 1].timestep == 0:
				transition[t] = blank_trans  # If future frame has timestep 0
			else:
				transition[t] = self.transitions.get(idx - self.history + 1 + t)

		for t in range(self.history, self.history + self.n):  # e.g. 4 5 6
			if transition[t - 1].nonterminal:
				transition[t] = self.transitions.get(idx - self.history + 1 + t)
			else:
				transition[t] = blank_trans  # If prev (next) frame is terminal

		return transition


	"""Returns a valid sample from a segment"""
	def _get_sample_from_segment(self, segment, i):
		valid = False
		while not valid:
			# Uniformly sample an element from within a segment
			sample = np.random.uniform(i * segment, (i + 1) * segment)  

			# Retrieve sample from tree with un-normalised probability
			prob, idx, tree_idx = self.transitions.find(sample)  
			
			# Resample if transition straddled current index or probablity 0
			if (self.transitions.index - idx) % self.capacity > self.n and (idx - self.transitions.index) % self.capacity >= self.history and prob != 0:
				valid = True  # Note that conditions are valid but extra conservative around buffer index 0

		# Retrieve all required transition data (from t - h to t + n)
		transition = self._get_transition(idx)

		# Create un-discretised state and nth next state
		state = torch.stack([trans.state for trans in transition[:self.history]]).to(device=self.device).to(dtype=torch.float32).div_(255)
		next_state = torch.stack([trans.state for trans in transition[self.n:self.n + self.history]]).to(device=self.device).to(dtype=torch.float32).div_(255)

		# Discrete action to be used as index
		action = torch.tensor([transition[self.history - 1].action], dtype=torch.int64, device=self.device)

		# Calculate truncated n-step discounted return R^n = Σ_k=0->n-1 (γ^k)R_t+k+1 (note that invalid nth next states have reward 0)
		R = torch.tensor([sum(self.discount ** n * transition[self.history + n - 1].reward for n in range(self.n))], dtype=torch.float32, device=self.device)

		# Mask for non-terminal nth next states
		nonterminal = torch.tensor([transition[self.history + self.n - 1].nonterminal], dtype=torch.float32, device=self.device)

		return prob, idx, tree_idx, state, action, R, next_state, nonterminal


	def sample(self, batch_size):
		# Retrieve sum of all priorities (used to create a normalised probability distribution)
		p_total = self.transitions.total()  

		# Batch size number of segments, based on sum over all probabilities
		segment = p_total / batch_size  

		# Get batch of valid samples
		batch = [self._get_sample_from_segment(segment, i) for i in range(batch_size)]  
		probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals = zip(*batch)
		states, next_states, = torch.stack(states), torch.stack(next_states)
		actions, returns, nonterminals = torch.cat(actions), torch.cat(returns), torch.stack(nonterminals)

		# Calculate normalised probabilities
		probs = np.array(probs, dtype=np.float32) / p_total  
		capacity = self.capacity if self.transitions.full else self.transitions.index

		# Compute importance-sampling weights w
		weights = (capacity * probs) ** -self.priority_weight  

		# Normalise by max importance-sampling weight from batch
		weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=self.device)  
		return tree_idxs, states, actions, returns, next_states, nonterminals, weights


	def update_priorities(self, idxs, priorities):
		priorities = np.power(priorities, self.priority_exponent)
		[self.transitions.update(idx, priority) for idx, priority in zip(idxs, priorities)]


	"""Set up internal state for iterator"""
	def __iter__(self):
		self.current_idx = 0
		return self


	"""Return valid states for validation"""
	def __next__(self):
		if self.current_idx == self.capacity:
			raise StopIteration

		# Create stack of states
		state_stack = [None] * self.history
		state_stack[-1] = self.transitions.data[self.current_idx].state
		prev_timestep = self.transitions.data[self.current_idx].timestep

		for t in reversed(range(self.history - 1)):
			if prev_timestep == 0:
				# If future frame has timestep 0
				state_stack[t] = blank_trans.state  
			else:
				state_stack[t] = self.transitions.data[self.current_idx + t - self.history + 1].state
				prev_timestep -= 1

		# Agent will turn into batch
		state = torch.stack(state_stack, 0).to(dtype=torch.float32, device=self.device).div_(255)  
		self.current_idx += 1
		return state