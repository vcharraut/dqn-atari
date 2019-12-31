import gym
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import time

from playground.utils.wrapper import wrap_environment
from playground.utils.memory import ReplayMemory
from playground.utils.model import CNN

import time


PATH_LOG = 'playground/atari/fig/log.txt'

class DQN():

	"""
	Initiale the Gym environnement BreakoutNoFrameskip-v4.
	The learning is done by a DQN.
	"""
	def __init__(self, env, config):

		# Gym environnement
		self.env = wrap_environment(env)

		# Parameters
		self.gamma = config.gamma
		self.bath_size = config.batch_size
		self.step_target_update = config.target_update
		self.epsilon_decay = config.epsilon_decay
		self.epsilon_start = config.epsilon_start
		self.epsilon_end = config.epsilon_end
		self.num_episodes = config.num_episodes
		self.start_learning = config.start_learning

		# List to save the rewards 
		self.plot_reward = []

		# Experience-Replay
		self.memory = ReplayMemory(config.memory_capacity)
		
		# CNN to compute the q-values
		self.model = CNN(self.env.observation_space.shape, self.env.action_space.n)

		# CNN to compute the q-target
		self.qtarget = CNN(self.env.observation_space.shape, self.env.action_space.n)

		# Backpropagation function
		# self.__optimizer = torch.optim.Adam(self.model.parameters(),
		#									 lr=learning_rate)
		self.__optimizer =  torch.optim.RMSprop(self.model.parameters(),
											lr=config.learning_rate,
											eps=0.001,
											alpha=0.95,
											momentum=0.95)

		# Error function
		self.__loss_fn = torch.nn.MSELoss()

		# Make the model using the GPU if available
		use_cuda = torch.cuda.is_available()
		if use_cuda:
			self.model.cuda()
			self.qtarget.cuda()
			self.device = torch.device('cuda')

	
	"""
	Compute the probabilty of exploration during the training
	using a e-greedy method with a decay.
	"""
	def act(self, state, step):
		# Compute the exploration rate
		eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
						math.exp(-1. * step / self.epsilon_decay)

		# Choice between exploration or intensification
		if np.random.rand() < eps_threshold:
			return self.env.action_space.sample(), eps_threshold
		else:
			state = torch.from_numpy(state).float() \
				.unsqueeze(0).to(self.device)
			return self.model(state).argmax().item(), eps_threshold


	"""
	Train the model.
	"""
	def learn(self, clone):

		# Skip if the memory is not full enough
		if self.memory.size < self.bath_size:
			return

		# Clone the q-values model to the q-targets model
		if clone:
			self.qtarget.load_state_dict(self.model.state_dict())

		# Get a random batch from the memory
		state, action, next_state, rewards, done = self.memory.sample(self.bath_size)

		state = torch.from_numpy(state).to(self.device)
		action = torch.from_numpy(action).long().to(self.device).unsqueeze(-1)
		next_state = torch.from_numpy(next_state).to(self.device)
		rewards = torch.from_numpy(rewards).to(self.device)
		done = torch.from_numpy(done).to(self.device)

		# Q values predicted by the model 
		pred = self.model(state).gather(1, action).squeeze()

		# Expected Q values are estimated from actions which gives maximum Q value
		max_q_target = self.qtarget(next_state).max(1).values.detach()

		# Apply Bellman equation
		y = rewards + (1. - done) * self.gamma * max_q_target

		# loss is measured from error between current and newly expected Q values
		loss = self.__loss_fn(y, pred)

		# backpropagation of loss to NN
		self.__optimizer.zero_grad()
		loss.backward()
		self.__optimizer.step()


	"""
	Save the model.
	"""
	def save_model(self):
		path = 'playground/atari/save/model.pt'
		torch.save(self.model.state_dict(), path)


	"""
	Write logs into a file
	"""
	def log(self, string):
		with open(PATH_LOG, "a") as f:
			f.write(string)
			f.write("\n")


	"""
	Run n episode to train the model.
	"""
	def train(self, display=False):
		step, previous_live = 0, 5
	
		

		for t in range(self.num_episodes + 1):
			episode_reward = 0.0
			done = False
			state = self.env.reset()
			start_time = time.time()

			# Run one episode until termination
			while not done:
				if display:
					self.env.render()

				# Select one action
				action, eps = self.act(state, step)

				# Get the output of env from this action
				next_state, r, _, live = self.env.step(action)

				# End the episode when the agent loses a life
				if previous_live is not live['ale.lives']:
					previous_live = live['ale.lives']
					done = True
				
				# Push the output to the memory
				self.memory.push(state, action, next_state, r, done)

				# Learn
				if step >= self.start_learning:
					if not step % self.step_target_update:
						self.learn(clone=True)
					else:
						self.learn(clone=False)

				step += 1
				state = next_state
				episode_reward += r


	
			self.log("[{}/{}] -- step:{} -- reward:{} -- eps:{} -- time:{}".format(
					t,
					self.num_episodes,
					step,
					episode_reward,
					round(eps, 3),
					round(time.time() - start_time, 4)))

			if not t % 10:
				mean_reward = sum(self.plot_reward[-10:]) / 10
				print("[{}/{}] -- step:{} -- avg_reward:{} -- eps:{} -- time:{}".format(
					t,
					self.num_episodes,
					step, mean_reward,
					round(eps, 3),
					round(time.time() - start_time, 4)))

			# Update the log and reset the env and variables
			self.env.reset()
			self.plot_reward.append(episode_reward)
			done = False

		self.env.close()


	"""
	Plot the rewards during the training.
	"""
	def figure(self):

		fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9, 9])
		window = 30
		rolling_mean = pd.Series(self.plot_reward).rolling(window).mean()
		std = pd.Series(self.plot_reward).rolling(window).std()
		ax1.plot(rolling_mean)
		ax1.fill_between(range(len(self.plot_reward)), rolling_mean -
						std, rolling_mean+std, color='orange', alpha=0.2)
		ax1.set_xlabel('Episode')
		ax1.set_ylabel('Reward')

		ax2.plot(self.plot_reward)
		ax2.set_xlabel('Episode')
		ax2.set_ylabel('Reward')

		fig.tight_layout(pad=2)
		# plt.show()
		plt.savefig('playground/atari/fig/run1.png')

