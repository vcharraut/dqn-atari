from playground.atari.config import Config
from playground.atari.algorithm.dqn import DQN


config = Config()
agent = DQN('BreakoutNoFrameskip-v4', config)
agent.train(True)
agent.figure()
agent.save_model()


