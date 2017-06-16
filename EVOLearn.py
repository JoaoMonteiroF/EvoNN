import pickle
import scipy.io

import numpy as np

import torch

import models_zoo
from Optimizer import Optimizer, DEOptimizer, SGDOptimizer, NewDe 
from Utils import buildAndSaveModels, data_loader
from models_zoo import MLP_MNIST, CNN

############# Import data set

(x_train, y_train), (x_valid, y_valid) = data_loader('mnist')

def main():

############# Create a Keras model and pass it to instantiate an optimizer

	numberOfEpochs = 1000
	popSize = 256

	model = MLP_MNIST()
	#model = CNN()

	if torch.cuda.is_available():
		model.cuda()

	#optimizer = DEOptimizer(x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid, preDefinedModel=model, n_epochs=numberOfEpochs, popSize = popSize, loss_function = 'cross_entropy')
	#optimizer = SGDOptimizer(x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid, preDefinedModel=model, n_epochs=numberOfEpochs, popSize = popSize, loss_function = 'cross_entropy')
	optimizer = NewDe(x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid, preDefinedModel=model, n_epochs=numberOfEpochs, popSize = popSize, loss_function = 'cross_entropy', cr=0.8, f=0.8, ub=10., lb=-10.)

	optimizer.modelFit()

if __name__ == "__main__":
	main()
