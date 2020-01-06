import gym
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import time
import glob
from tqdm import tqdm

from playground.utils.wrapper import make_atari, wrap_deepmind
from playground.utils.memory import ReplayMemory
from playground.utils.model import CNN, Dueling_CNN


class DQN():

	"""
	Initiale the Gym environnement BreakoutNoFrameskip-v4.
	The learning is done by a DQN.
	"""
	def __init__(self, env, config, doubleq, dueling, evaluation=False, record=False):

		# Gym environnement
		self.env = wrap_deepmind(make_atari(env))

		if record:
			self.env = gym.wrappers.Monitor(
				self.env, 'playground/atari/recording/dqn', force=True)

		# Are we in evaluation mode ? 
		self._evaluation = evaluation
		
		if not evaluation : 
			# Parameters
			self.gamma = config.gamma
			self.bath_size = config.batch_size
			self.step_target_update = config.target_update
			self.freq_learning = config.freq_learning
			self.epsilon_decay = config.epsilon_decay
			self.epsilon_start = config.epsilon_start
			self.epsilon_end = config.epsilon_end
			self.num_steps = config.num_steps
			self.start_learning = config.start_learning

			# Experience-Replay
			self.memory = ReplayMemory(config)

		# Architecture parameters
		self.doubleq = doubleq

		# List to save the rewards 
		self.plot_reward = []
		self.plot_eval = []

		# Architecture of the neural networks
		if dueling:
			use_dueling = '_dueling'
			self.model = Dueling_CNN(self.env.observation_space.shape, self.env.action_space.n)
			if not evaluation : 
				self.qtarget = Dueling_CNN(self.env.observation_space.shape, self.env.action_space.n)
		else:
			use_dueling = ''
			self.model = CNN(self.env.observation_space.shape, self.env.action_space.n)
			if not evaluation : 
				self.qtarget = CNN(self.env.observation_space.shape, self.env.action_space.n)


		# Backpropagation function
		self.__optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

		# Error function
		self.__loss_fn = torch.nn.SmoothL1Loss(reduction='mean')

		# Make the model using the GPU if available
		if torch.cuda.is_available():
			self.model.cuda()
			if not evaluation: self.qtarget.cuda()
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cpu')


		if doubleq:
			use_doubleq = '_doubleq'
		else:
			use_doubleq = ''

		# Path to the logs folder
		specs = use_doubleq  + use_dueling

		# See if training has been made with this configuration
		specs += '_' + str(len(glob.glob1('playground/atari/log/', 'dqn' + specs + '*.txt')) + 1)

		# Path for the saves
		self.path_log = 'playground/atari/log/dqn' + specs + '.txt'
		self.path_save = 'playground/atari/save/dqn' + specs
		self.path_fig = 'playground/atari/fig/dqn' + specs
		if not evaluation:
			config.save_config('playground/atari/log/dqn-config' + specs + '.txt', env)


	"""
	Get the action for the qvalue given a state
	"""
	def get_policy(self, state):
		with torch.no_grad():
			state = torch.from_numpy(state).float() \
					.unsqueeze(0).to(self.device)
			return self.model(state).argmax().item()
		

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
			return self.get_policy(state), eps_threshold


	"""
	Train the model.
	"""
	def learn(self):

		# Get a random batch from the memory
		state, action, next_state, rewards, done = self.memory.sample()

		# Q values predicted by the model 
		pred = self.model(state).gather(1, action).squeeze()

		with torch.no_grad():
			# Expected Q values are estimated from actions which gives maximum Q value
			if self.doubleq:
				action_by_qvalue = self.model(next_state).argmax(1).long().unsqueeze(-1)
				max_q_target = self.qtarget(next_state).gather(1, action_by_qvalue).squeeze()
			else:
				max_q_target = self.qtarget(next_state).argmax(1)

		# Apply Bellman equation
		y = rewards + (1. - done) * self.gamma * max_q_target

		# Loss is measured from error between current and newly expected Q values
		loss = self.__loss_fn(y, pred)

		# Backpropagation of loss to NN
		self.__optimizer.zero_grad()
		loss.backward()
		for param in self.model.parameters():
			param.grad.data.clamp_(-1, 1)
		self.__optimizer.step()


	"""
	Save the model.
	"""
	def save_model(self, final=False):
		if final:
			path = self.path_save + '-final.pt'
		else:
			path = self.path_save + '.pt'

		torch.save(self.model.state_dict(), path)


	"""
	Write logs into a file
	"""
	def log(self, string):
		with open(self.path_log, "a") as f:
			f.write(string + "\n")



	"""
	Evaluate the model during the training
	"""
	def evaluation(self, num_episodes=30):

		self.model.eval()
		sum_reward = 0.0
		for _ in range(num_episodes):
			episode_reward = 0.0
			done = False
			state = self.env.reset()
			for _ in range(10):
				state, _, done, _ = self.env.step(0)
				if done:
					state = self.env.reset()

			while not done:
				action = self.get_policy(state)

				# Get the output of env from this action
				state, reward, done, _ = self.env.step(action)

				episode_reward += reward

			sum_reward += episode_reward
			
		self.plot_eval.append(sum_reward)
		mean = sum_reward / num_episodes
		self.log("## EVALUATION --  avg_reward:{}".format(mean))

		self.model.train()

			
	"""
	Run n episode to train the model.
	"""
	def train(self, display=False):
		step, episode, best = 0, 0, 0
		step_evaluation = 20000

		pbar = tqdm(total=self.num_steps)
	
		while step <= self.num_steps:
			episode_reward = 0.0
			done = False
			state = self.env.reset()
			start_time = time.time()
			episode += 1

			# Run one episode until termination
			while not done:
				if display:
					self.env.render()

				# Select one action
				action, eps = self.act(state, step)

				# Get the output of env from this action
				next_state, reward, done, _ = self.env.step(action)

				# Push the output to the memory
				self.memory.push(next_state, action, reward, done)

				# Learn
				if step >= self.start_learning:
					if not step % self.freq_learning:
						self.learn()

					# Clone the q-values model to the q-targets model
					if not step % self.step_target_update:
						self.qtarget.load_state_dict(self.model.state_dict())

					if not step % step_evaluation:
						self.evaluation()

				step += 1
				pbar.update()
				episode_reward += reward
				state = next_state

			end_time = round(time.time() - start_time, 4)

			if not episode % 20:
				mean_reward = sum(self.plot_reward[-20:]) / 20
				max_reward = max(self.plot_reward[-20:])
				if max_reward > best:
					self.log("Saving model, best reward :{}".format(max_reward))
					self.save_model()
					best = max_reward
				self.log("Episode {} -- step:{} -- avg_reward:{} -- best_reward:{} -- eps:{} -- time:{}".format(
					episode,
					step,
					mean_reward,
					max_reward,
					round(eps, 3),
					end_time))

			if not episode % 5:
				self.plot_reward.append(episode_reward)

		pbar.close()
		self.env.close()
		self.save_model(final=True)
		self.figure()


	"""
	Eval a trained model for n episodes.
	"""
	def test(self, num_episodes=50, display=False, model_path=None):

		if self._evaluation:
			if model_path is None:
				raise ValueError('No path model given.')
			self.model = copy.deepcody(torch.load(model_path))
	
		self.model.eval()
		self.plot_reward.clear()

		for episode in range(1, num_episodes + 1):
			# Run one episode until termination
			episode_reward = 0
			done = False
			state = self.env.reset()
			while not done:
				if display:
					self.env.render()

				action = self.get_policy(state)

				# Get the output of env from this action
				state, reward, done, _ = self.env.step(action)

				episode_reward += reward


			print("Episode {} -- reward:{} ".format(episode, episode_reward))
			self.plot_reward.append(episode_reward)

		self.env.close()
		self.figure(train=False)


	"""
	Plot the rewards during the training.
	"""
	def figure(self, train=True):
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
		ax2.plot(self.plot_eval)
		ax2.set_xlabel('Episode')
		ax2.set_ylabel('Reward')

		fig.tight_layout(pad=2)

		if train:
			path = self.path_fig + '.png'
		else:
			path = self.path_fig + '-eval.png'
			plt.show()
		plt.savefig(path)
		
