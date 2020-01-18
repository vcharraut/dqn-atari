import torch
from random import sample
from collections import deque


class ReplayMemory():

    def __init__(self, config, device):
        self.capacity = config.memory_capacity
        self.memory = deque(maxlen=self.capacity)
        self.device = device
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
            self.next_states[t] = torch.from_numpy(
                self.memory[i][0].__array__()).to(dtype=torch.float32)
            if i != 0:
                self.states[t] = torch.from_numpy(
                    self.memory[i-1][0].__array__()).to(dtype=torch.float32)
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
