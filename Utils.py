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
		pickle.dump(modelToSave, open('$/scratch/nwv-632-aa/Models/bestModel-'+str(index)+'.p', 'wb'))

		index+=1

def buildAndSaveModelsFromHof(hallOfFame, model):
	index = 1
	for ind in hallOfFame:
		updateParameters(ind)
		pickle.dump(model, open('/RQexec/joaobmf/Models/bestModel'+str(index)+'.p', 'wb')
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

def batch_generator(X, y, batch_size=32):
	
	number_of_batches = int(np.ceil(data_size/batch_size))
	data_size = X.size()[0]
	
	while True:

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

		return (torch.from_numpy(x_train), torch.from_numpy(y_train)), (torch.from_numpy(x_valid), torch.from_numpy(y_valid))

	except dataSetException:
		print 'The required data set is not avaliable for load. Value passed:', lossFuncException.value	

if __name__ == "__main__":
	plot_fitness('valid_fitness.p')
