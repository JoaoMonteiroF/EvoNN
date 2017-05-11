from datetime import datetime
import time
import numpy as np
import os
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error

class lossFuncException(Exception):
	def __init__(self, value):
		self.value=value
	def __str__(self):
		return repr(self.value)

def buildAndSaveModels(optimizer):

	if not os.path.isdir('Models'):
		os.makedirs('Models')

	dateTime = datetime.now().strftime('%d-%m-%Y %H:%M')

	k = len(optimizer.bestIndividuals)

	results = [0]*k

	index=0

	for individual in optimizer.bestIndividuals:

		fitness=optimizer.testModel(np.asarray(individual))

		modelResults = [index+1, -1.0*optimizer.model.loss, optimizer.model.delayForMinCorr]

		results[index] = modelResults

		modelToSave = optimizer.model.EVOModel

		modelToSave.save('Models/bestModel-'+str(index+1)+'-'+dateTime+'.hdf5')

		index+=1

	f=open('DEOutputs-'+dateTime+'.txt','w')
	for output in results:
		f.write(str(output)+'\n')
	f.close

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
