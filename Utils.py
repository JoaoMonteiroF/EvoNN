from datetime import datetime
import time
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from keras import datasets
from keras.utils import to_categorical
from keras import backend as K

class lossFuncException(Exception):
	def __init__(self, value):
		self.value=value
	def __str__(self):
		return repr(self.value)

class dataSetException(Exception):
	def __init__(self, value):
		self.value=value
	def __str__(self):
		return repr(self.value)

def buildAndSaveModels(optimizer):
	index = 1
	for individual in optimizer.bestIndividuals:

		optimizer.model.updateParameters(individual)
		modelToSave = optimizer.model.EVOModel
		pickle.dump(modelToSave, open('$/scratch/nwv-632-aa/Models/bestModel-'+str(index)+'.p', 'wb'))

		index+=1

def buildAndSaveModelsFromHof(hallOfFame, model):
	index = 1
	for ind in hallOfFame:
		updateParameters(ind)
		pickle.dump(model, open('/RQexec/joaobmf/Models/bestModel'+str(index)+'.p', 'wb'))
		index+=1

def countParameters(myModel):
	counter = 0
	for param in myModel.parameters():
		acc =1
		for value in param.size():
			acc*=value
		counter+=acc

	return counter

def tensorElementsCount(myTensor):
	acc =1

	for value in myTensor.size():
		acc*=value

	return acc

def calculateLoss(y_pred, y_true, lossFunction):
	try:
		if lossFunction is 'mse':
			return F.pairwise_distance(y_pred, y_true,  p=2)
		elif lossFunction is 'msa':
			return F.pairwise_distance(y_pred, y_true,  p=1)
		elif lossFunction is 'cross_entropy':
			return F.cross_entropy(y_pred, y_true)
		else:
			raise Exception(lossFunction)
	except lossFuncException:
		print 'Wrong loss function definition. Value passed:', lossFuncException.value

def plot_fitness(pkl = 'fitness.p'):
	to_plot = pickle.load(file(pkl))
	plt.plot(to_plot)
	plt.legend('Fitness')	
	plt.show()

def find_last_improvement(fitness_list):
	last_fitness = fitness_list[-1]

	for i, value in enumerate(reversed(fitness_list)):
		if value>last_fitness:
			print('here')
			return i
	return i

def batch_generator(X, y, batch_size=32):
	
	data_size = X.size()[0]
	number_of_batches = int(np.ceil(data_size/batch_size))

	for i in xrange(0, number_of_batches):
		inputs_batch = X[i*batch_size:min((i+1)*batch_size, data_size)]
		targets_batch = y[i*batch_size:min((i+1)*batch_size, data_size)]
		
		yield (inputs_batch, targets_batch)

def data_loader(dataSetName):
	try:
		if dataSetName is 'mnist':
			(x_train, y_train), (x_valid, y_valid) = datasets.mnist.load_data()
			img_rows, img_cols = 28, 28
			num_classes = 10
		elif dataSetName is 'cifar10':
			(x_train, y_train), (x_valid, y_valid) = datasets.cifar10.load_data()
			img_rows, img_cols = 32, 32
			num_classes = 10
		elif dataSetName is 'cifar100':
			(x_train, y_train), (x_valid, y_valid) = datasets.cifar100.load_data()
			img_rows, img_cols = 32, 32
			num_classes = 100
		elif dataSetName is 'boston':
			(x_train, y_train), (x_valid, y_valid) = datasets.boston_housing.load_data()
			return (x_train, y_train), (x_valid, y_valid)
		else:
			raise Exception(dataSetName)

		x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
		x_valid = x_valid.reshape(x_valid.shape[0], 1, img_rows, img_cols)


		x_train = x_train.astype('Float32')
		x_valid = x_valid.astype('Float32')

		x_train /= 255
		x_valid /= 255

		y_train = y_train.astype('Int64')
		y_valid = y_valid.astype('Int64')

		return (torch.from_numpy(x_train), torch.from_numpy(y_train)), (torch.from_numpy(x_valid), torch.from_numpy(y_valid))

	except dataSetException:
		print 'The required data set is not avaliable for load. Value passed:', lossFuncException.value	

if __name__ == "__main__":
	plot_fitness('valid_fitness.p')
