import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Dense_nn(nn.Module):
	
	def __init__(self, in_dim, out_dim, hidden_layer):
		super(Dense_nn, self).__init__()
		
		self.fc1 = nn.Linear(in_dim, hidden_layer)
		# self.bn1 = nn.BatchNorm2d(hidden_layer)
		self.fc2 = nn.Linear(hidden_layer, hidden_layer)
		# self.bn2 = nn.BatchNorm2d(hidden_layer)
		self.fc3 = nn.Linear(hidden_layer, out_dim)
			

	def forward(self, x):
		# x = self.bn1(self.fc1(x))
		x = self.fc1(x)
		x = F.relu(x)
		# x = self.bn2(self.fc2(x))
		x = self.fc2(x)
		x = F.relu(x)
		return self.fc3(x)


class Conv2d_nn(nn.Module):
	
	def __init__(self,
				in_dim,
				hidden_layer,
				out_dim, 
				kernel_size=5):
		super(Conv2d_nn, self).__init__()
		
		# self.dropout = dropout

		self.conv1 = nn.Conv2d(1, 16, kernel_size)
		self.conv2 = nn.Conv2d(16, 32, kernel_size)
		self.conv3 = nn.Conv2d(32, 64, kernel_size)
		self.fc1 = nn.Linear(3136, hidden_layer)
		self.fc2 = nn.Linear(hidden_layer, out_dim)
			

	def forward(self, x): 
		x = x.view(-1, 1, 84, 84)
		# conv1
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		# x = F.dropout(x, p=self.dropout, training=training)
		# conv2
		x = F.relu(F.max_pool2d(self.conv2(x), 2))
		# x = F.dropout(x, p=self.dropout, training=training)
		#conv3
		x = F.relu(F.max_pool2d(self.conv3(x), 2))
		# x = F.dropout(x, p=self.dropout, training=training)

		x = x.view(-1, x.shape[2] * x.shape[3] * 64)
		# fc1
		x = self.fc1(x)
		x = F.relu(x)
		return self.fc2(x)
