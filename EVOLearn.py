import cPickle
import scipy.io

import numpy as np

import models_zoo
from Optimizer import Optimizer, DEOptimizer, NNEVO 
from Utils import buildAndSaveModels
from models_zoo import MLP

#from scoop import futures

def main():

############# Import data set

	x_train = scipy.io.loadmat('feat_train_arousal.mat') # Modulation feature set for arousal
	y_train = scipy.io.loadmat('tgt_train_arousal.mat') # Labels set for arousal
	x_valid = scipy.io.loadmat('feat_dev_arousal.mat')
	y_valid = scipy.io.loadmat('tgt_dev_arousal.mat')

	#x_train = scipy.io.loadmat('feat_train_valence.mat') # Modulation feature set for valence
	#y_train = scipy.io.loadmat('tgt_train_valence.mat') # Labels set for arousal
	#x_valid = scipy.io.loadmat('feat_dev_valence.mat')
	#y_valid = scipy.io.loadmat('tgt_dev_valence.mat')

############# Create a Keras model and pass it to instantiate an optimizer

	numberOfEpochs = 1000
	popSize = 250

	model = MLP()

	optimizer = DEOptimizer(x_train=x_train['features'], y_train=y_train['gs_dev'], x_valid=x_valid['features'], y_valid=y_valid['gs_dev'], preDefinedModel=model, n_epochs=numberOfEpochs, popSize = popSize, loss = 'mse')

	optimizer.modelFit()

	buildAndSaveModels(optimizer)

if __name__ == "__main__":
	main()
