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
        state = torch.from_numpy(state[-1])
        state = state.to(dtype=torch.uint8, device=torch.device('cpu'))
        self.memory.append((state, action, reward, terminal))


    def get_states(self, index_ns):
        state, next_state = torch.empty(4, 84, 84), torch.empty(4, 84, 84)
        index_s = index_ns - 1

        next_state[-1] = self.memory[index_ns][0].to(dtype=torch.float32)
        state[-1] = self.memory[index_s][0].to(dtype=torch.float32)

        for i in range(1, 4):
            if index_ns-i >= 0:
                next_state[3-i] = self.memory[index_ns-i][0].to(dtype=torch.float32)
            else: 
                next_state[3-i] = torch.zeros(84, 84)

            if index_s-i >= 0:
                state[3-i] = self.memory[index_s-i][0].to(dtype=torch.float32)
            else: 
                state[3-i] = torch.zeros(84, 84)

        return state, next_state


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
            self.states[t], self.next_states[t] = self.get_states(i)
            self.actions[t] = self.memory[i][1]
            self.rewards[t] = self.memory[i][2]
            self.terminals[t] = self.memory[i][3]

        return (self.states,
                self.actions.unsqueeze(-1),
                self.next_states,
                self.rewards,
                self.terminals)
