from playground.atari.config import Config
from playground.atari.algorithm.dqn import DQN
import argparse

parser = argparse.ArgumentParser(description='DQN')
parser.add_argument('--doubleq', type=int, help='')
parser.add_argument('--dueling', type=int, help='')
parser.add_argument('--adam', type=int, help='')
parser.add_argument('--mse', type=int, help='')

args = parser.parse_args()
config = Config()
agent = DQN('BreakoutNoFrameskip-v4', config,
                                    doubleq=args.doubleq,
                                    dueling=args.dueling,
                                    adam=args.adam,
                                    mse=args.mse)
agent.train()
agent.figure()
agent.test()
agent.figure(train=False)




