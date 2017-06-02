from datetime import datetime
import time
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error
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
		modelToSave.save('$/scratch/nwv-632-aa/Models/bestModel-'+str(index)+'.h5')

		index+=1

def buildAndSaveModelsFromHof(hallOfFame):
	index = 1
	for ind in hallOfFame:
		updateParameters(ind)
		model.save('/RQexec/joaobmf/Models/bestModel'+str(index)+'.h5')
		index+=1

def tensorElementsCount(myTensor):

	tensorShape = myTensor.shape
	accum = 1

	for element in tensorShape:
		accum *= element

	return accum

def countParameters(model):

	model.summary()
	totalParameters = 0
	layersToPrint = model.layers
	for layer in layersToPrint:
		paramsList = layer.get_weights()

		for params in paramsList:
			try:
				totalParameters += tensorElementsCount(params)
			except AttributeError:
				totalParameters += len(params)

	return totalParameters

def calculateLoss(y_true, y_pred, lossFunction):
	try:
		if lossFunction is 'mse':
			return mean_squared_error(y_true, y_pred)
		elif lossFunction is 'msa':
			return mean_absolute_error(y_true, y_pred)
		elif lossFunction is 'cross_entropy':
			return log_loss(y_true, y_pred)
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

		if K.image_data_format() == 'channels_first':
			x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
			x_valid = x_valid.reshape(x_valid.shape[0], 1, img_rows, img_cols)
			input_shape = (1, img_rows, img_cols)
		else:
			x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
			x_valid = x_valid.reshape(x_valid.shape[0], img_rows, img_cols, 1)
			input_shape = (img_rows, img_cols, 1)


		x_train = x_train.astype('float32')
		x_valid = x_valid.astype('float32')
		x_train /= 255
		x_valid /= 255
		print('x_train shape:', x_train.shape)
		print(x_train.shape[0], 'train samples')
		print(x_valid.shape[0], 'test samples')

		# convert class vectors to binary class matrices
		y_train = to_categorical(y_train, num_classes)
		y_valid = to_categorical(y_valid, num_classes)

		return (x_train, y_train), (x_valid, y_valid)

	except dataSetException:
		print 'The required data set is not avaliable for load. Value passed:', lossFuncException.value	

if __name__ == "__main__":
	plot_fitness('valid_fitness.p')
