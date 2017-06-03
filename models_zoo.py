from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNN(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, 10)

	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
		x = x.view(-1, 320)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)
		return F.softmax(x)


class MLP_MNIST(nn.Module):
	def __init__(self):
		super(MLP_MNIST, self).__init__()
		self.den1 = nn.Linear(784, 128)
		self.den2 = nn.Linear(128, 10)

	def forward(self, x):
		x = x.view(x.numel())
		x = self.den1(x)
		x = F.dropout(x)
		x = F.relu(x)
		x = self.den2(x)
		return F.softmax(x)

def CNN_MNIST():
	def __init__(self):
		super(CNN_MNIST, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3)
		self.conv2 = nn.Conv2d(32, 64, 3)
		self.den1 = nn.Linear(9216, 128)
		self.den2 = nn.Linear(128, 10)

	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.max_pool2d(x, 2)
		x = F.dropout(x)
		x = F.relu(x)
		x = x.view(x.numel())
		x = self.den1(x)
		x = F.dropout(x)
		x = F.relu(x)
		x = self.den2(x)
		x = F.relu(x)
		return F.softmax(x)

