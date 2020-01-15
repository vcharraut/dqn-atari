import torch

from core.utils.memory import ReplayMemory
from core.utils.model import CNN
from core.atari.algorithm.base import Base


class DQN(Base):

    """
    Initiale the Gym environnement BreakoutNoFrameskip-v4.
    The learning is done by a DQN.
    """

    def __init__(self,
                 env,
                 config,
                 train,
                 record):
        super().__init__(env, config, train, record)

        # Experience-Replay
        self.memory = ReplayMemory(config, self.device)

        # Architecture of the neural networks
        self.model = CNN(self.env.observation_space.shape,
                         self.env.action_space.n)
        if train:
            self.qtarget = CNN(
                self.env.observation_space.shape, self.env.action_space.n)

        # Backpropagation function
        self.__optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate)

    """
    Train the model.
    """

    def learn(self):

        # Get a random batch from the memory
        state, action, next_state, rewards, done = self.memory.sample()

        # Q values predicted by the model
        pred = self.model(state).gather(1, action).squeeze()

        with torch.no_grad():
            # Expected Q values are estimated from actions
            # which gives maximum Q value

            action_by_qvalue = self.model(
                next_state).argmax(1).long().unsqueeze(-1)
            max_q_target = self.qtarget(next_state).gather(
                1, action_by_qvalue).squeeze()

        # Apply Bellman equation
        y = rewards + (1. - done) * self.gamma * max_q_target

        # Loss is measured from error between current and newly
        # expected Q values
        loss = self._Base__loss_fn(y, pred)

        # Backpropagation of loss to NN
        self.__optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.__optimizer.step()
