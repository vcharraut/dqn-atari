import gym
import torch

from playground.atari.config import Config
from playground.atari.algorithm.dqn import DQN


config = Config()
agent = DQN('BreakoutNoFrameskip-v4', config)

agent.model.load_state_dict(torch.load('playground/atari/save/dqn_huberloss_no.pt'))
agent.model.cuda()
agent.model.eval()

for t in range(20):
	# Run one episode until termination
	r = 0
	done = False
	state = agent.env.reset()
	while not done:
		agent.env.render()

		state = torch.from_numpy(state).float()
		state = state.unsqueeze(0).to(torch.device('cuda'))
		action = agent.model(state).argmax().item()

		# Get the output of env from this action
		state, reward, done, _ = agent.env.step(action)
		r += reward

	print(t, r)

agent.env.close()
