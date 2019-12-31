import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
		self._num_actions = num_actions

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