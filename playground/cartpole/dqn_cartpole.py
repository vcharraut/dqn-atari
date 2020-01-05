import gym
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


from playground.utils.memory import CartpoleMemory
from playground.utils.model import Dense_NN


PATH_LOG = 'playground/cartpole/fig/log.txt'


class DQN_Cartpole():

	"""
	Initiale the Gym environnement Cartpole-v1.
	The learning is done by a DQN.
	Params : 
		- learning_rate
		- hidden_layer
		- gamma
		- batch_sizes
		- step_target_update
	"""

	def __init__(self,
				learning_rate,
				hidden_layer,
				gamma, 
				batch_size,
				step_target_update,
				evaluation=False,
				record=False):
		# Gym environnement Cartpole
		self.env = gym.make('CartPole-v1')

		if record:
			self.env = gym.wrappers.Monitor(
				self.env, 'playground/cartpole/recording/', force=True)
		

		self.evaluation = evaluation
		# List to save the rewards 
		self.plot_reward = []

		# Parameters
		if not evaluation:
			self.learning_rate = learning_rate
			self.hidden_layer = hidden_layer
			self.gamma = gamma
			self.bath_size = batch_size
			self.step_target_update = step_target_update

			self.solved = False
			self.episode_done = []

			# Experience-Replay buffer
			self.memory = CartpoleMemory(50000)

		# Dense neural network to compute the q-values
		self.model = Dense_NN(in_dim=self.env.observation_space.shape[0],
							out_dim=self.env.action_space.n,
							hidden_layer=hidden_layer)

		if not evaluation:
			# Dense neural network to compute the q-target
			self.q_target_nn = Dense_NN(in_dim=self.env.observation_space.shape[0],
								out_dim=self.env.action_space.n,
								hidden_layer=hidden_layer)

			# Backpropagation function
			self.__optimizer = torch.optim.RMSprop(
				self.model.parameters(), lr=learning_rate)

			# Error function
			self.__loss_fn = torch.nn.MSELoss(reduction='mean')

		# Make the model using the GPU if available
		if torch.cuda.is_available():
			self.model.cuda('cuda')
			if not evaluation:
				self.q_target_nn.cuda('cuda')
			self.device = torch.device('cuda')
		else: 
			self.device = torch.device('cpu')

	"""
	Get an action of the max qvalue from the model.
	"""

	def get_policy(self, state):
		with torch.no_grad():
			x = torch.from_numpy(state).float().to(torch.device('cuda'))
			return self.model(x).argmax().item()


	"""
	Compute the probabilty of exploration during the training
	using a e-greedy method with a decay.
	"""
	def act(self, state, step):
		# Compute the exploration rate
		eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * step / EPS_DECAY)

		# Random choice between exploration or intensification
		if np.random.rand() < eps_threshold:
			return self.env.action_space.sample(), eps_threshold
		else:
			return self.get_policy(state), eps_threshold

	"""
	Train the model.
	"""

	def learn(self, clone):

		# Skip if the memory is not full enough
		if self.memory.size < self.bath_size:
			return

		# Clone the q-values model to the q-targets model
		if clone:
			self.q_target_nn.load_state_dict(self.model.state_dict())

		# Get a random batch from the memory
		state, action, next_state, rewards, done = self.memory.sample(self.bath_size)

		state = torch.from_numpy(state).to(self.device)
		action = torch.from_numpy(action).to(self.device).long().unsqueeze(-1)
		next_state = torch.from_numpy(next_state).to(self.device)
		rewards = torch.from_numpy(rewards).to(self.device)
		done = torch.from_numpy(done).to(self.device)

		# Q values predicted by the model
		pred = self.model(state).gather(1, action).squeeze()

		with torch.no_grad():
			# Expected Q values are estimated from actions which gives maximum Q value
			max_q_target = self.q_target_nn(next_state).max(1).values

			# Apply Bellman equation
			y = rewards + (1. - done) * self.gamma * max_q_target

		# loss is measured from error between current and newly expected Q values
		loss = self.__loss_fn(y, pred)
		self.list_loss.append(loss.item())

		# backpropagation of loss to NN
		self.__optimizer.zero_grad()
		loss.backward()
		for param in self.model.parameters():
			param.grad.data.clamp_(-1, 1)
		self.__optimizer.step()

		return round(loss.item(), 4)

	"""
	Save the model.
	"""

	def save_model(self):
		path = 'playground/cartpole/save/lr={}_hidden={}_gamma={}_batchsize={}_steptarget={}.pt'.format(
                    self.learning_rate,
                    self.hidden_layer,
                    self.gamma,
                    self.bath_size,
                    self.step_target_update)
		torch.save(self.model.state_dict(), path)

	"""
	Write a log into a file
	"""

	def log(self, string):
		with open(PATH_LOG, "a") as f:
			f.write(string)
			f.write("\n")

	"""
	Run n episode with randoms choices to feed the memory.
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
				self.memory.push(state, action, next_state, reward, done)
				state = next_state

			# Update the log and reset the env and variables
			self.env.reset()
			done = False


	"""
	Run n episode to train the model.
	"""
	def train(self, display=False):
		sum_reward, step, mean = 0, 0, 0
		mean_reward = []
		done = False

		state = self.env.reset()

		for t in range(EPISODE_LEARN + 1):
			# Run one episode until termination
			while not done:
				if display:
					self.env.render()

				# Select one action
				action, eps = self.act(state, step)

				# Get the output of env from this action
				next_state, reward, done, info = self.env.step(action)

				# Update the values for the log
				sum_reward += reward
				step += 1

				# Add the output to the memory
				self.memory.push(state, action, next_state, reward, done)

				# Learn
				if step % self.step_target_update == 0:
					loss = self.learn(clone=True)
				else:
					loss = self.learn(clone=False)

				state = next_state

			# Compute the mean reward for the last 5 actions
			# If the reward is close to the max (500), we stop
			# the training
			mean_reward.append(sum_reward)
			if t % 10 == 0:
				mean = sum(mean_reward) / 10
				if mean >= self.env.spec.reward_threshold:
					self.episode_done = t
					print("[{}/{}], r:{}, avg:{}, loss:{}, eps:{}".format(
                                            t, EPISODE_LEARN, sum_reward, mean, loss, eps))
					break
				else:
					mean_reward.clear()

			if t % 20 == 0:
				print("[{}/{}], r:{}, avg:{}, loss:{}, eps:{}".format(
					t, EPISODE_LEARN, sum_reward, mean, loss, round(eps, 3)))

			# Update the log and reset the env and variables
			self.env.reset()
			self.plot_reward_train.append(sum_reward)
			sum_reward = 0
			done = False

		self.episode_done = t
		self.env.close()

	"""
	Run the trained the model and verify on 100 episodes if the
	Cartpole environnement is solved.
	(Solved : Mean Reward => 475)
	"""

	def test(self, display=True):
		sum_reward = 0
		done = False
		state = self.env.reset()
		self.plot_reward.clear()

		for _ in range(EPISODE_PLAY):
			# Run one episode until termination
			while not done:
				if display:
					self.env.render()

				# Select one action
				action = self.get_policy(state)

				# Get the output of env from this action
				state, reward, done, _ = self.env.step(action)

				# Update the values for the log
				sum_reward += reward

			# Update the log and reset the env and variables
			self.env.reset()
			self.plot_reward.append(sum_reward)
			sum_reward = 0
			done = False

		mean = sum(self.plot_reward) / len(self.plot_reward)
		if mean >= self.env.spec.reward_threshold:
			self.solved = True
			self.save_model()
			print("## Solved after {} episodes.".format(self.episode_done))
		else:
			print('## Not solved, mean={} '.format(mean))
		print("## Params: LR={}, Gamma={}, Hidden_layer={}, Batch_size={}, Step_target_update={}".format(
                    self.learning_rate, self.gamma, self.hidden_layer, self.bath_size, self.step_target_update
                ))
		print('#' * 85)

		self.env.close()

	"""
	Plot the rewards and the loss during the training.
	Plot the rewards only during the play.
	The figures are saved as file.
	"""

	def play(self, num_episodes=20, display=False, model_path=None):
	
		if self.evaluation:
			if model_path is None:
				raise ValueError('No path model given.')
			self.model.load_state_dict(torch.load(model_path))
	
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

		

	def figure(self, training, save=True):

		if self.solved:
			path = 'playground/cartpole/fig/solved/'
		else:
			path = 'playground/cartpole/fig/notsolved/'

		if training:
			plot_reward = self.plot_reward_train
			path += 't_lr={}_hidden={}_gamma={}_batchsize={}_steptarget={}.png'.format(
                            self.learning_rate,
                            self.hidden_layer,
                            self.gamma,
                            self.bath_size,
                            self.step_target_update)
			fig, ((ax1), (ax2), (ax3)) = plt.subplots(3, 1, figsize=[9, 9])
			rolling_mean = pd.Series(self.list_loss).rolling(50).mean()
			std = pd.Series(self.list_loss).rolling(50).std()
			ax3.plot(rolling_mean)
			ax3.fill_between(range(len(self.list_loss)), rolling_mean -
                            std, rolling_mean+std, color='orange', alpha=0.2)
			ax3.set_xlabel('Episode')
			ax3.set_ylabel('Loss')
		else:
			plot_reward = self.plot_reward_play
			path += 'p_lr={}_hidden={}_gamma={}_batchsize={}_steptarget={}.png'.format(
                            self.learning_rate,
                            self.hidden_layer,
                            self.gamma,
                            self.bath_size,
                            self.step_target_update)
			fig, ((ax1), (ax2)) = plt.subplots(2, 1, figsize=[9, 9])

		window = 30
		rolling_mean = pd.Series(plot_reward).rolling(window).mean()
		std = pd.Series(plot_reward).rolling(window).std()
		ax1.plot(rolling_mean)
		ax1.fill_between(range(len(plot_reward)), rolling_mean -
                   std, rolling_mean+std, color='orange', alpha=0.2)
		ax1.set_xlabel('Episode')
		ax1.set_ylabel('Reward')

		ax2.plot(plot_reward)
		ax2.set_xlabel('Episode')
		ax2.set_ylabel('Reward')

		fig.tight_layout(pad=2)
		if save:
			plt.savefig(path)
		else:
			plt.show()

