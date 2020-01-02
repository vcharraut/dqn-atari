import gym
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import time
import glob

from playground.utils.wrapper import wrap_environment
from playground.utils.memory import PrioritizedReplayMemory
from playground.utils.model import RainbowNetwork


class Rainbow():

	"""
	Initiale the Gym environnement BreakoutNoFrameskip-v4.
	The learning is done by a Rainbow.
	"""
	def __init__(self, env, config, adam):

		# Gym environnement
		self.env = wrap_environment(env)

		# Parameters
		self.gamma = config.gamma
		self.batch_size = config.batch_size
		self.step_target_update = config.target_update
		self.freq_learning = config.freq_learning
		self.epsilon_decay = config.epsilon_decay
		self.epsilon_start = config.epsilon_start
		self.epsilon_end = config.epsilon_end
		self.num_steps = config.num_steps
		self.start_learning = config.start_learning
		self.vmin = config.vmin
		self.vmax = config.vmax
		self.prior_expo = config.prior_expo
		self.prior_samp = config.prior_samp
		self.n = config.multi_step
		self.atoms = config.atoms

		# List to save the rewards 
		self.plot_reward = []

		# Experience-Replay
		self.memory = PrioritizedReplayMemory(config.memory_capacity)
		
		# Dueling CNN for the qvalues and qtarget
		self.model = RainbowNetwork(self.env.observation_space.shape,
									self.env.action_space.n,
									config.atoms,
									config.noisy_nets,
									config.architecture)
		self.qtarget = RainbowNetwork(self.env.observation_space.shape,
									self.env.action_space.n,
									config.atoms,
									config.noisy_nets,
									config.architecture)

		# Support (range) of z
		self.support = torch.linspace(config.vmin, config.vmax, config.atoms).to(torch.device('cuda'))  
		self.delta_z = (config.vmax - config.vmin) / (config.atoms - 1)
		

		# Backpropagation function
		if adam:
			optim_method = '_adam'
			self.__optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, eps=config.adam_exp)
		else:
			optim_method = '_rmsprop'
			self.__optimizer =  torch.optim.RMSprop(self.model.parameters(),
		 									lr=config.learning_rate,
		 									eps=0.001,
		 									alpha=0.95,
		 									momentum=0.95)

		# Make the model using the GPU if available
		use_cuda = torch.cuda.is_available()
		if use_cuda:
			self.model.cuda()
			self.qtarget.cuda()
			self.device = torch.device('cuda')

		# Path to the logs folder
		specs = optim_method 
		# See if training has been made with this configuration
		specs += '_' + str(len(glob.glob1('playground/atari/log/', 'rainbow' + specs + '*.txt')) + 1)

		self.path_log = 'playground/atari/log/rainbow' + specs + '.txt'
		self.path_save = 'playground/atari/save/rainbow' + specs + '.pt'
		self.path_fig = 'playground/atari/fig/rainbow' + specs + '.png'
		config.save_config('playground/atari/log/rainbow' + specs + '-config.txt', env)


	"""
	Get the action for the qvalue given a state
	"""
	def get_policy(self, state):
		with torch.no_grad():
			state = torch.from_numpy(state).to(self.device)
			return (self.model(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()
	

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
	def learn(self, clone):

		# Clone the q-values model to the q-targets model
		if clone:
			self.qtarget.load_state_dict(self.model.state_dict())

		# Get a random batch from the memory
		idxs, states, actions, returns, next_states, nonterminals, weights = self.memory.sample(self.batch_size)

		# Q values predicted by the model 
		prob_state = self.model(states, log=True)
		prob_state_action = prob_state[range(self.batch_size), actions]

		with torch.no_grad():
			# Expected Q values are estimated from actions which gives maximum Q value
			qvalue_next_state = self.model(next_states)
			distri_qvalue = self.support.expand_as(qvalue_next_state) * qvalue_next_state
			actions_next_state = distri_qvalue.sum(2).argmax(1)
			self.qtarget.reset_noise()

			prob_next_state = self.qtarget(next_states)
			prob_next_state_action = prob_next_state[range(self.batch_size), actions_next_state]

			# Compute Tz (Bellman operator T applied to z)
			# Tz = R^n + (γ^n)z (accounting for terminal states)
			Tz = returns.unsqueeze(1) + nonterminals * (self.gamma ** self.n) * self.support.unsqueeze(0) 

			# Clamp between supported values
			Tz = Tz.clamp(min=self.vmin, max=self.vmax)  

			# Compute L2 projection of Tz onto fixed support z
			# b = (Tz - Vmin) / Δz
			b = (Tz - self.vmin) / self.delta_z  

			l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
			# Fix disappearing probability mass when l = b = u (b is int)
			l[(u > 0) * (l == u)] -= 1
			u[(l < (self.atoms - 1)) * (l == u)] += 1

			# Distribute probability of Tz
			m = states.new_zeros(self.batch_size, self.atoms)
			offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
			# m_l = m_l + p(s_t+n, a*)(u - b)
			m.view(-1).index_add_(0, (l + offset).view(-1), (prob_next_state_action * (u.float() - b)).view(-1))  
			# m_u = m_u + p(s_t+n, a*)(b - l)
			m.view(-1).index_add_(0, (u + offset).view(-1), (prob_next_state_action * (b - l.float())).view(-1))  

		# Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
		loss = -torch.sum(m * prob_state_action, 1)  
		self.__optimizer.zero_grad()

		# Backpropagate importance-weighted minibatch loss
		(weights * loss).mean().backward()  
		self.__optimizer.step()

		# Update priorities of sampled transitions
		self.memory.update_priorities(idxs, loss.detach().cpu().numpy())  


	"""
	Save the model.
	"""
	def save_model(self):
		torch.save(self.model.state_dict(), self.path_save)


	"""
	Write logs into a file
	"""
	def log(self, string):
		with open(self.path_log, "a") as f:
			f.write(string + "\n")


	"""
	Run n episode to train the model.
	"""
	def train(self, display=False):
		priority_weight_increase = (1 - self.prior_samp) / (self.num_steps - self.start_learning)
		step, episode, previous_live = 0, 0, 5
		best = 0.0
	
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
				action = int(action)
				action = 3 if action > 3 else action

				# Get the output of env from this action
				next_state, reward, _, live = self.env.step(action)

				# End the episode when the agent loses a life
				if previous_live is not live['ale.lives']:
					previous_live = live['ale.lives']
					done = True
				
				# Push the output to the memory
				self.memory.append(state, action, reward, done)

				if not step % self.freq_learning:
					self.model.reset_noise()

				# Learn
				if step >= self.start_learning:
					self.memory.priority_weight = min(self.prior_samp + priority_weight_increase, 1)
					if not step % self.freq_learning:
						if not step % self.step_target_update:
							self.learn(clone=True)
						else:
							self.learn(clone=False)

				step += 1
				episode_reward += reward
				state = next_state


			end_time = round(time.time() - start_time, 4)

			if not episode % 50:
				mean_reward = sum(self.plot_reward[-50:]) / 50
				max_reward = max(self.plot_reward[-50:])
				self.log("Episode {} -- step:{} -- avg_reward:{} -- best_reward:{} -- eps:{} -- time:{}".format(
					episode,
					step,
					mean_reward,
					max_reward,
					round(eps, 3),
					end_time))

			if episode_reward > best:
				self.log("Saving model, best reward :{}".format(episode_reward))
				self.save_model()
				best = episode_reward

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
		plt.savefig(self.path_fig)

