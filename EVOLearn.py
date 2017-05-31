from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error

import numpy as np

import models_zoo

from Utils import buildAndSaveModelsFromHof, countParameters, tensorElementsCount, data_loader
from models_zoo import MLP_MNIST

from deap import base
from deap import creator
from deap import tools

from scoop import futures

from itertools import chain

import random
import array
import pickle
import os.path

############# Import data set

(x_train, y_train), (x_valid, y_valid) = data_loader('mnist')

############# Define Model

model = MLP_MNIST()

############# Evaluation related functions

def calculateLoss(y_true, y_pred, lossFunction):

		if lossFunction is 'mse':
			return mean_squared_error(y_true, y_pred)
		elif lossFunction is 'msa':
			return mean_absolute_error(y_true, y_pred)
		elif lossFunction is 'cross_entropy':
			return log_loss(y_true, y_pred)
		else:
			print('Undefined loss function')
			return 'err'

def updateParameters(parameters):

	parametersCopy = parameters

	layersList = model.layers

	for layerToUpdate in layersList:

		paramsToUpdate = []
		paramsList = layerToUpdate.get_weights()

		for params in paramsList:

			try:
				shapeWeights = params.shape
				numberOfParameters = tensorElementsCount(params)
				newParameters = parametersCopy[0:numberOfParameters]
				newParameters = newParameters.reshape(shapeWeights)
		
			except AttributeError:
				numberOfParameters = len(params)
				newParameters = np.asarray(parametersCopy[0:numberOfParameters])

			paramsToUpdate.append(newParameters)

			parametersCopy = np.delete(parametersCopy, range(numberOfParameters))


		layerToUpdate.set_weights(paramsToUpdate)

def evaluate(parameters):

	updateParameters(np.asarray(parameters))
	
	output = model.predict(x_train)

	loss = calculateLoss(y_train, output, lossFunction='cross_entropy')

	return [loss,];

def testModel(parameters):

	updateParameters(np.asarray(parameters))
	
	output = model.predict(x_valid)

	loss = calculateLoss(y_valid, output, lossFunction='cross_entropy')

	return [loss,];

def computeValidationFitness(hallOfFame):

	bestFitness = float('inf')

	for ind in hallOfFame:
		fitness = testModel(ind)
		if fitness < bestFitness:
			bestFitness = fitness
	return bestFitness

############# DE functions

def mutDE(y, a, b, c, f):
	size = len(y)
	for i in range(len(y)):
		y[i] = a[i] + f*(b[i]-c[i])
	return y

def cxBinomial(x, y, cr):
	size = len(x)
	index = random.randrange(size)
	for i in range(size):
		if i == index or random.random() < cr:
		    x[i] = y[i]
	return x

def cxExponential(x, y, cr):
	size = len(x)
	index = random.randrange(size)
	# Loop on the indices index -> end, then on 0 -> index
	for i in chain(range(index, size), range(0, index)):
		x[i] = y[i]
		if random.random() < cr:
		    break
	return x

############ 

NDIM = countParameters(model)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
#		creator.create("Individual", array.array, typecode='f', fitness=creator.FitnessMin)
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -10, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, NDIM)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mutate", mutDE, f=0.8)
toolbox.register("mate", cxExponential, cr=0.8)
toolbox.register("select", tools.selRandom, k=3)
toolbox.register("evaluate", evaluate) 
toolbox.register("map", futures.map) 

if __name__ == "__main__":

############# Evolutionary Algorithm

	print('Start of evolution...')

	MU = 256
	NGEN = 1000
	patience = 30
	runMax = 300
	epoch=0
	run=0
	found = False
	CPNamePOP = None

	#Look for checkpoint populations

	for i in range(runMax,-1,-1):
		for j in range(NGEN,-1,-1):	
			epochStr = str(j)
			runStr = str(i)

			if (os.path.exists('/RQexec/joaobmf/CP/pop-'+runStr+'-'+epochStr+'.p')):
				CPNamePOP = '/RQexec/joaobmf/CP/pop-'+runStr+'-'+epochStr+'.p'
				epoch=j+1
				run=i+1
				found=True
				break
		if found:
			break

	if found:
		pop = pickle.load(open(CPNamePOP, 'rb'))
		g=epoch
	else:
		pop = toolbox.population(n=MU)
		g=1

	if (os.path.exists('train_fitness.p') and os.path.exists('valid_fitness.p') and found):
		bestTrainFitnessHist=pickle.load(open('train_fitness.p', 'rb'))
		bestValidFitnessHist=pickle.load(open('valid_fitness.p', 'rb'))
	else:
		bestTrainFitnessHist = []
		bestValidFitnessHist = []

	hof = tools.HallOfFame(10)
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", np.mean)
	stats.register("std", np.std)
	stats.register("min", np.min)
	stats.register("max", np.max)

	logbook = tools.Logbook()
	logbook.header = "gen", "evals", "std", "min", "avg", "max"

	# Evaluate the individuals
	print('Evaluating individuals...')
	fitnesses = toolbox.map(toolbox.evaluate, pop)
	for ind, fit in zip(pop, fitnesses):
		ind.fitness.values = fit


	print('Start of generations...')

	record = stats.compile(pop)
	logbook.record(gen=0, evals=len(pop), **record)
	print(logbook.stream)

	iterationsWithoutImprovement = 0
	currentBestValidationFitness = 0
	lastBestValidationFitness = float('inf')

	while (g<=NGEN and iterationsWithoutImprovement <= patience):
		children = []
		for agent in pop:
			# We must clone everything to ensure independence
			a, b, c = [toolbox.clone(ind) for ind in toolbox.select(pop)]
			x = toolbox.clone(agent)
			y = toolbox.clone(agent)
			y = toolbox.mutate(y, a, b, c)
			z = toolbox.mate(x, y)
			del z.fitness.values
			children.append(z)

		fitnesses = toolbox.map(toolbox.evaluate, children)
		for (i, ind), fit in zip(enumerate(children), fitnesses):
			ind.fitness.values = fit
			if ind.fitness > pop[i].fitness:
				pop[i] = ind

		hof.update(pop)
		record = stats.compile(pop)
		logbook.record(gen=g, evals=len(pop), **record)

		currentBestValidationFitness = computeValidationFitness(hof)

		if currentBestValidationFitness <= lastBestValidationFitness:
			lastBestValidationFitness = currentBestValidationFitness
			iterationsWithoutImprovement = 0
		else:
			iterationsWithoutImprovement += 1

		bestTrainFitnessHist.append(hof[0].fitness.values[0])
		bestValidFitnessHist.append(currentBestValidationFitness)

		pickle.dump(bestTrainFitnessHist, open('train_fitness.p', 'wb'))
		pickle.dump(bestValidFitnessHist, open('valid_fitness.p', 'wb'))

		pickle.dump(pop, open('/RQexec/joaobmf/CP/pop-'+str(run)+'-'+str(g)+'.p', 'wb'))
		
		g+=1

	buildAndSaveModelsFromHof(hof)
	print("Best fitness found is:", hof[0].fitness.values[0])
	pickle.dump(logbook, open('DEOptimizer_logbook.p', 'wb'))
