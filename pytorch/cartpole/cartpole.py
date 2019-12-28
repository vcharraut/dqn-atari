import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import pandas as pd


# Exploration parameters
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

# Training parameters
EPISODE_WARMUP = 2000
EPISODE_LEARN = 301
EPISODE_PLAY = 101

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor


class Network(nn.Module):
	
	def __init__(self, in_dim, out_dim, hidden_layer):
		super(Network, self).__init__()
		
		self.fc1 = nn.Linear(in_dim, hidden_layer)
		self.fc2 = nn.Linear(hidden_layer, hidden_layer)
		self.fc3 = nn.Linear(hidden_layer, out_dim)
			

	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		return self.fc3(x)



class Cartpole:

	def __init__(self,
				learning_rate,
				hidden_layer, 
				gamma, 
				batch_size, 
				step_target_update):
		# Gym environnement Cartpole
		self.env = env = gym.make('CartPole-v1')

		# Parameters
		self.learning_rate = learning_rate
		self.hidden_layer = hidden_layer
		self.gamma = gamma
		self.bath_size = batch_size
		self.step_target_update = step_target_update

		# List to save the rewards and the loss
		# throughout the training
		self.plot_reward = []
		self.list_loss = []

		# List for the Experience-Replay
		# and aximum capacity for the Experience-Replay
		self.memory = []
		self.capacity = 100000
		
		# Dense neural network to compute the q-values
		self.q_nn = Network(in_dim=self.env.observation_space.shape[0], 
							out_dim=env.action_space.n,
							hidden_layer=hidden_layer)

		# Dense neural network to compute the q-target
		self.q_target_nn = Network(in_dim=self.env.observation_space.shape[0], 
							out_dim=env.action_space.n,
							hidden_layer=hidden_layer)

		# Backpropagation function
		#self.__optimizer = torch.optim.Adam(self.q_nn.parameters(),
		#									lr=learning_rate)
		self.__optimizer =  torch.optim.RMSprop(self.q_nn.parameters(), lr=learning_rate)

		# Error function
		self.__loss_fn = torch.nn.MSELoss()

		# Make the model using the GPU if available
		if use_cuda:
			self.q_nn.cuda('cuda')
			self.q_target_nn.cuda('cuda')

	

	"""
	Update the memory
	"""
	def update_memory(self, state, action, next_state, reward, end_step):

		# If the memory is full, we delete the first element
		if len(self.memory) > self.capacity:
			del self.memory[0]

		# Add the information to the memory
		self.memory.append((FloatTensor([state]),
							LongTensor([action]),
							FloatTensor([next_state]),
							FloatTensor([reward]),
							FloatTensor([end_step])))


	"""
	Return a random sample from self.memory of len = number
	"""
	def sample_batch(self, number):
		sample = random.sample(self.memory, number)

		# Unwrap the batch to get the variables
		state, action, next_state, reward, done = zip(*sample)

		# Transform the variables to fit the computation
		state = torch.stack(state).squeeze()
		action = torch.stack(action)
		next_state = torch.stack(next_state).squeeze()
		reward = torch.stack(reward).squeeze()
		done = torch.stack(done).squeeze()

		return state, action, next_state, reward, done


	"""
	Get an action of the max qvalue from the q_nn
	"""
	def qvalue(self, state):
		x = torch.from_numpy(state).float()
		x = x.to(torch.device('cuda'))
		return self.q_nn(x).argmax().item()

	"""
	Compute the probabilty of exploration during the training
	using a e-greedy method with a decay
	"""
	def act(self, state, step):
		# Compute the exploration rate
		eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    	math.exp(-1. * step / EPS_DECAY)

		# Random choice between exploration or intensification
		if np.random.rand() < eps_threshold:
			return self.env.action_space.sample()
		else:
			return self.qvalue(state)


	"""
	Train the model
	"""
	def learn(self, clone):

		# Skip if the memory is not full enough
		if len(self.memory) < self.bath_size:
			return

		# Clone the q-values model to the q-targets model
		if clone:
			self.q_target_nn.load_state_dict(self.q_nn.state_dict())

		# Get a random batch from the memory
		state, action, next_state, rewards, done = self.sample_batch(self.bath_size)

		# Q values predicted by the model 
		pred = self.q_nn(state).gather(1, action).squeeze()

		# Expected Q values are estimated from actions which gives maximum Q value
		next_q_values = self.q_target_nn(next_state).detach()
		max_q_next_target = torch.max(next_q_values, 1).values

		# Apply Bellman equation
		y = rewards + (1. - done) * self.gamma * max_q_next_target


		# loss is measured from error between current and newly expected Q values
		loss = self.__loss_fn(y, pred)
		#loss = F.smooth_l1_loss(y, pred)
		self.list_loss.append(loss)

		# backpropagation of loss to NN
		self.__optimizer.zero_grad()
		loss.backward()
		self.__optimizer.step()


	"""
	Run n episode with randoms choices to feed the memory
	"""
	def warmup(self):
		done = False
		state = self.env.reset()

		for _ in range(EPISODE_WARMUP):
			# Run one episode until termination
			while not done:
				# Take one random action
				action = self.env.action_space.sample()

				# Get the output of env from this action
				next_state, reward, done, info = self.env.step(action)

				# Add the output to the memory
				self.update_memory(state, action, next_state, reward, done)
				state = next_state

			# Update the log and reset the env and variables
			self.env.reset()
			done = False

		print("Size of memory : ", len(self.memory))


	"""
	Run n episode to train the model
	"""
	def train(self, display=False):
		sum_reward, step, mean = 0, 0, 0
		mean_reward = []
		done = False

		state = self.env.reset()

		for t in range(EPISODE_LEARN):
			# Run one episode until termination
			while not done:
				if display:
					self.env.render()

				# Select one action
				action = self.act(state, step)

				# Get the output of env from this action
				next_state, reward, done, info = self.env.step(action)

				# Update the values for the log
				sum_reward += reward
				step += 1

				#if step >= 350: reward = 5

				#if done: reward = 0.1
				

				# Add the output to the memory
				self.update_memory(state, action, next_state, reward, done)

				# Learn
				if step % self.step_target_update == 0:
					self.learn(clone=True)
				else:
					self.learn(clone=False)

				state = next_state


			# Compute the mean reward for the last 5 actions
			# If the reward is close to the max (500), we stop
			# the training
			mean_reward.append(sum_reward)


			if t % 20 == 0:
				mean = sum(mean_reward) / 20
				print("[{}/{}], r:{}, avg:{}".format(t, EPISODE_LEARN, sum_reward, mean))
				if mean >= 480:
					break
				else:
					mean_reward.clear()
			
			# Update the log and reset the env and variables
			self.env.reset()
			self.plot_reward.append(sum_reward)
			sum_reward = 0
			done = False

		self.env.close()


	"""
	Run the trained the model
	"""
	def play(self, display=True):
		self.plot_reward = []
		sum_reward, step = 0, 0
		done = False

		state = self.env.reset()

		for _ in range(EPISODE_PLAY):
			# Run one episode until termination
			while not done:
				if display:
					self.env.render()

				# Select one action
				action = self.qvalue(state)

				# Get the output of env from this action
				state, reward, done, _ = self.env.step(action)

				# Update the values for the log
				sum_reward += reward
				step += 1

			# Update the log and reset the env and variables
			self.env.reset()
			self.plot_reward.append(sum_reward)
			sum_reward = 0
			done = False

		self.env.close()


	"""
	Plot the log
	"""
	def log(self, training):
		if self.plot_reward == None:
			print('No log available')
		else:
			window = 50
			fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9, 9])
			rolling_mean = pd.Series(self.plot_reward).rolling(window).mean()
			std = pd.Series(self.plot_reward).rolling(window).std()
			ax1.plot(rolling_mean)
			ax1.fill_between(range(len(self.plot_reward)), rolling_mean -
							std, rolling_mean+std, color='orange', alpha=0.2)
			ax1.set_title(
				'Episode Length Moving Average ({}-episode window)'.format(window))
			ax1.set_xlabel('Episode')
			ax1.set_ylabel('Episode Length')

			ax2.plot(self.plot_reward)
			ax2.set_title('Episode Length')
			ax2.set_xlabel('Episode')
			ax2.set_ylabel('Episode Length')

			fig.tight_layout(pad=2)
			if training:
				plt.savefig(
					't_lr={}_hidden={}_gamma={}_batchsize={}_steptarget={}.png'
					.format(self.learning_rate,
							self.hidden_layer,
							self.gamma,
							self.bath_size,
							self.step_target_update))
			else:
				plt.savefig(
					'p_lr={}_hidden={}_gamma={}_batchsize={}_steptarget={}.png'
					.format(self.learning_rate,
							self.hidden_layer,
							self.gamma,
							self.bath_size,
							self.step_target_update))



	def log_loss(self):
		window = 50
		fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9, 9])
		rolling_mean = pd.Series(self.list_loss).rolling(window).mean()
		std = pd.Series(self.list_loss).rolling(window).std()
		ax1.plot(rolling_mean)
		ax1.fill_between(range(len(self.list_loss)), rolling_mean -
						std, rolling_mean+std, color='orange', alpha=0.2)
		ax1.set_title(
			'Episode Length Moving Average ({}-episode window)'.format(window))
		ax1.set_xlabel('Episode')
		ax1.set_ylabel('Episode Length')

		ax2.plot(self.list_loss)
		ax2.set_title('Episode Length')
		ax2.set_xlabel('Episode')
		ax2.set_ylabel('Episode Length')

		fig.tight_layout(pad=2)
		plt.show()


	def run(self):
		# self.warmup()
		self.train(display=True)
		self.log(training=True)
		self.play(display=True)
		self.log(training=False)


hyperparams = []
hyperparams.append([1e-3, 64, 0.99, 64, 50])
"""
hyperparams.append([1e-3, 32, 0.9, 32, 1000])
hyperparams.append([1e-3, 64, 0.99, 64, 1000])
hyperparams.append([1e-3, 64, 0.99, 64, 500])
hyperparams.append([1e-3, 64, 0.8, 64, 500])
"""



for p in hyperparams:
	agent = Cartpole(p[0], p[1], p[2], p[3], p[4])
	agent.run()

