from playground.atari.config import Config
from playground.atari.algorithm.dqn import DQN
import argparse

parser = argparse.ArgumentParser(description='DQN')
parser.add_argument('--doubleq', type=int, default=1, help='')
parser.add_argument('--dueling', type=int, default=1, help='')

args = parser.parse_args()
config = Config()
agent = DQN('BreakoutNoFrameskip-v4', config,
                                    doubleq=args.doubleq,
                                    dueling=args.dueling)
agent.train()
agent.figure()
agent.test()
agent.figure(train=False)




