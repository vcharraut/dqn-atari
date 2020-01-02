import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class Dense_NN(nn.Module):

	def __init__(self, in_dim, out_dim, hidden_layer):
		super(Dense_NN, self).__init__()

		self.fc = nn.Sequential(
			nn.Linear(in_dim, hidden_layer),
			nn.ReLU(),
			nn.Linear(hidden_layer, hidden_layer),
			nn.ReLU(),
			nn.Linear(hidden_layer, out_dim)
		)

	def forward(self, x):
		return self.fc(x)


class CNN(nn.Module):

	def __init__(self, input_shape, num_actions):
		super(CNN, self).__init__()
		self._input_shape = input_shape

		self.features = nn.Sequential(
			nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1),
			nn.ReLU()
		)

		self.fc = nn.Sequential(
			nn.Linear(self.feature_size, 512),
			nn.ReLU(),
			nn.Linear(512, num_actions)
		)

	def forward(self, x):
		x = self.features(x).view(x.size()[0], -1)
		return self.fc(x)

	@property
	def feature_size(self):
		x = self.features(torch.zeros(1, *self._input_shape))
		return x.view(1, -1).size(1)


class Dueling_CNN(nn.Module):
	def __init__(self, input_shape, num_actions):
		super(Dueling_CNN, self).__init__()
		self.num_actions = num_actions

		self.features = nn.Sequential(
			nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1),
			nn.ReLU()
		)

		self.fc1_adv = nn.Linear(in_features=7*7*64, out_features=512)
		self.fc1_val = nn.Linear(in_features=7*7*64, out_features=512)

		self.fc2_adv = nn.Linear(in_features=512, out_features=num_actions)
		self.fc2_val = nn.Linear(in_features=512, out_features=1)

		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.features(x).view(x.size()[0], -1)

		adv = self.relu(self.fc1_adv(x))
		val = self.relu(self.fc1_val(x))

		adv = self.fc2_adv(adv)
		val = self.fc2_val(val).expand(x.size(0), self.num_actions)

		x = val + adv - \
			adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
		return x


class NoisyLinear(nn.Module):

	def __init__(self, in_features, out_features, std_init=0.5):
		super(NoisyLinear, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.std_init = std_init
		self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
		self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
		self.register_buffer(
			'weight_epsilon', torch.empty(out_features, in_features))
		self.bias_mu = nn.Parameter(torch.empty(out_features))
		self.bias_sigma = nn.Parameter(torch.empty(out_features))
		self.register_buffer('bias_epsilon', torch.empty(out_features))
		self.reset_parameters()
		self.reset_noise()

	def reset_parameters(self):
		mu_range = 1 / math.sqrt(self.in_features)
		self.weight_mu.data.uniform_(-mu_range, mu_range)
		self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
		self.bias_mu.data.uniform_(-mu_range, mu_range)
		self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

	def _scale_noise(self, size):
		x = torch.randn(size)
		return x.sign().mul_(x.abs().sqrt_())

	def reset_noise(self):
		epsilon_in = self._scale_noise(self.in_features)
		epsilon_out = self._scale_noise(self.out_features)
		self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
		self.bias_epsilon.copy_(epsilon_out)

	def forward(self, input):
		if self.training:
			return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
		else:
			return F.linear(input, self.weight_mu, self.bias_mu)


class RainbowNetwork(nn.Module):

	def __init__(self, input_shape, action_space, atoms, noisy_std, architecture):
		super(RainbowNetwork, self).__init__()
		self.atoms = atoms
		self.action_space = action_space

		if architecture == 'canonical':
			hidden_layer = 512
			self.convs = nn.Sequential(
				nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
			  	nn.ReLU(),
				nn.Conv2d(32, 64, kernel_size=4, stride=2),
				nn.ReLU(),
				nn.Conv2d(64, 64, kernel_size=3, stride=1),
				nn.ReLU()
			)
			self.conv_output_size = 3136
		elif architecture == 'data-efficient':
			hidden_layer = 256
			self.convs = nn.Sequential(
				nn.Conv2d(input_shape[0], 32, kernel_size=5, stride=5),
				nn.ReLU(),
				nn.Conv2d(32, 64, kernel_size=5, stride=5),
				nn.ReLU()
			)
			self.conv_output_size = 576

		self.fc_h_v = NoisyLinear(
			self.conv_output_size, hidden_layer, std_init=noisy_std)
		self.fc_h_a = NoisyLinear(
			self.conv_output_size, hidden_layer, std_init=noisy_std)
		self.fc_z_v = NoisyLinear(hidden_layer, self.atoms, std_init=noisy_std)
		self.fc_z_a = NoisyLinear(
			hidden_layer, action_space * self.atoms, std_init=noisy_std)

	def forward(self, x, log=False):
		x = self.convs(x)
		x = x.view(-1, self.conv_output_size)

		# Value stream
		v = self.fc_z_v(F.relu(self.fc_h_v(x)))

		# Advantage stream
		a = self.fc_z_a(F.relu(self.fc_h_a(x)))
		v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)

		# Combine streams
		q = v + a - a.mean(1, keepdim=True)

		# Use log softmax for numerical stability
		if log:
			# Log probabilities with action over second dimension
			return F.log_softmax(q, dim=2)
		else:
			# Probabilities with action over second dimension
			return F.softmax(q, dim=2)

	def reset_noise(self):
		for name, module in self.named_children():
			if 'fc' in name:
				module.reset_noise()
