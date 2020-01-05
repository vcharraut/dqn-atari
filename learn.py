import argparse

from playground.atari.config import Config, ConfigRainbow
from playground.atari.algorithm.dqn import DQN
from playground.atari.algorithm.dqn import DQN
from playground.cartpole.dqn_cartpole import DQN_Cartpole

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

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='', help='')
parser.add_argument('--algo', type=str, default='', help='')
parser.add_argument('--display', type=int, default=1, help='')

display = args.display

if args.env == 'cartpole':
	agent = DQN_Cartpole(None, 8, None, None, None, evaluation=True)
	agent.play()
elif args.env == 'atari':
	if args.algo == 'dqn':
		config = Config()
		agent = DQN('BreakoutNoFrameskip-v4', config, False, False, False, True)
	elif args.algo == 'dqn+':
		config = Config()
		agent = DQN('BreakoutNoFrameskip-v4', config, True, True, False, True)
	elif args.algo == 'rainbow':
		config = ConfigRainbow()
		agent = Rainbow('BreakoutNoFrameskip-v4', config)
		agent.test(display=display, model_path='playground/atari/save/rainbow_adam_1.pt')
	else:
		raise ValueError('Algo is not valid')
else:
	raise ValueError('Environnement is not valid')