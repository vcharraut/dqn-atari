from playground.atari.config import ConfigRainbow
from playground.atari.algorithm.rainbow import Rainbow
import argparse


parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--minimal', type=int, help='')
parser.add_argument('--adam', type=int, help='')
args = parser.parse_args()

if args.minimal:
    config = ConfigRainbow(num_steps=100000,
                        target_update=2000,
                        start_learning=1600,
                        freq_learning=1,
                        learning_rate=0.0001,
                        architecture='data-efficient')
else:
    config = ConfigRainbow()

agent = Rainbow('BreakoutNoFrameskip-v4', config, adam=args.adam)
agent.train()
agent.figure()
agent.test()
agent.figure(train=False)


