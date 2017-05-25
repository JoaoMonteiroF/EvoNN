import cPickle
import scipy.io

import numpy as np

import models_zoo
from Optimizer import Optimizer, DEOptimizer, SGDOptimizer, NNEVO 
from Utils import buildAndSaveModels, data_loader
from models_zoo import MLP_MNIST

############# Import data set

(x_train, y_train), (x_valid, y_valid) = data_loader('mnist')

def main():

############# Create a Keras model and pass it to instantiate an optimizer

	numberOfEpochs = 1000
	popSize = 256

	model = MLP_MNIST()

	optimizer = DEOptimizer(x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid, preDefinedModel=model, n_epochs=numberOfEpochs, popSize = popSize, loss = 'cross_entropy')

	optimizer.modelFit()

	buildAndSaveModels(optimizer)

if __name__ == "__main__":
	main()
