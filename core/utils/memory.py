import torch
from random import sample
from collections import deque
import numpy as np


class CartpoleMemory():

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

        return (np.array(state, dtype=np.float32),
                np.array(action),
                np.array(reward, dtype=np.float32),
                np.array(next_state, dtype=np.float32),
                np.array(done, dtype=np.uint8))

    @property
    def size(self):
        return len(self.memory)


class ReplayMemory():

    def __init__(self, config):
        self.capacity = config.memory_capacity
        self.memory = deque(maxlen=self.capacity)
        self.device = torch.device('cuda')
        self.batch_size = config.batch_size

        self.states = torch.empty((self.batch_size, 4, 84, 84)).to(self.device)
        self.next_states = torch.empty(
            (self.batch_size, 4, 84, 84)).to(self.device)
        self.actions = torch.empty(self.batch_size).to(self.device)
        self.rewards = torch.empty(self.batch_size).to(self.device)
        self.terminals = torch.empty(self.batch_size).to(self.device)

    @property
    def size(self):
        return len(self.memory)

    """Add an experience to the memory"""

    def push(self, state, action, reward, terminal):
        state = torch.from_numpy(state)
        state = state.to(dtype=torch.uint8, device=torch.device('cpu'))
        self.memory.append((state, action, reward, terminal))

    """Return a random sample from self.memory of size = number"""

    def sample(self):
        sample_indices = sample(range(self.size), self.batch_size)

        self.states = torch.empty((self.batch_size, 4, 84, 84)).to(self.device)
        self.next_states = torch.empty(
            (self.batch_size, 4, 84, 84)).to(self.device)
        self.actions = torch.empty(self.batch_size).to(
            self.device).to(dtype=torch.long)
        self.rewards = torch.empty(self.batch_size).to(self.device)
        self.terminals = torch.empty(self.batch_size).to(self.device)

        for t in range(self.batch_size):
            i = sample_indices[t]
            self.next_states[t] = self.memory[i][0].to(dtype=torch.float32)
            if i != 0:
                self.states[t] = self.memory[i-1][0].to(dtype=torch.float32)
            else:
                self.states[t] = torch.zeros(84, 84)
            self.actions[t] = self.memory[i][1]
            self.rewards[t] = self.memory[i][2]
            self.terminals[t] = self.memory[i][3]

        return (self.states,
                self.actions.unsqueeze(-1),
                self.next_states,
                self.rewards,
                self.terminals)
