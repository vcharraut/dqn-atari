from playground.atari.config import Config
from playground.atari.algorithm.DQN import DQN


config = Config()
agent = DQN('BreakoutNoFrameskip-v4', config)
agent.train()
agent.figure()


