from playground.atari.config import Config
from playground.atari.algorithm.rainbow import Rainbow

config = Config()
agent = Rainbow('BreakoutNoFrameskip-v4', config, adam=1, mse=0)
agent.train()
agent.figure()
agent.save_model()




