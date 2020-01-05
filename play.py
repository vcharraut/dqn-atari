import argparse

from playground.atari.config import Config, ConfigRainbow
from playground.atari.algorithm.rainbow import Rainbow
from playground.atari.algorithm.dqn import DQN
from playground.cartpole.dqn_cartpole import DQN_Cartpole


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='', help='')
parser.add_argument('--algo', type=str, default='', help='')
parser.add_argument('--display', type=int, default=1, help='')
parser.add_argument('--record', type=int, default=0, help='')



args = parser.parse_args()

display = args.display
record = args.record

if args.env == 'cartpole':
	agent = DQN_Cartpole(None, 8, None, None, None, evaluation=True, record=record)
	agent.play(display=display, model_path='playground/cartpole/save/lr=0.01_hidden=8_gamma=0.99_batchsize=256_steptarget=100.pt')
elif args.env == 'atari':
	if args.algo == 'dqn':
		print("No model saved yet.")
		config = Config()
		agent = DQN('BreakoutNoFrameskip-v4', config, False, False, False, True, evaluation=True, record=record)
	elif args.algo == 'dqn+':
		print("No model saved yet.")
		config = Config()
		agent = DQN('BreakoutNoFrameskip-v4', config, True, True, False, True, evaluation=True, record=record)
	elif args.algo == 'rainbow':
		config = ConfigRainbow()
		agent = Rainbow('BreakoutNoFrameskip-v4', config, True, evaluation=True, record=record)
		agent.test(display=display, model_path='playground/atari/save/rainbow_adam_1-final.pt')
	else:
		raise ValueError('Algo is not valid')
else:
	raise ValueError('Environnement is not valid')

			
