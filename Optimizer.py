from deap import base
from deap import creator
from deap import tools

from keras.callbacks import EarlyStopping

import numpy as np

from itertools import chain

import random
import array
import pickle
import os.path

from Utils import countParameters, calculateLoss, lossFuncException, tensorElementsCount

class Optimizer(object):

	def __init__(self, x_train, y_train, x_valid, y_valid, preDefinedModel, n_epochs=100, popSize = 300, loss = 'mse'):

		self.numberOfEpochs = n_epochs
		self.PopulationSize = popSize
		self.x_train = x_train
		self.y_train = y_train
		self.x_valid = x_valid
		self.y_valid = y_valid
		self.loss = loss

		self.model = NNEVO(preModel = preDefinedModel, lossFunction = loss)

		self.totalNumberOfParameters = countParameters(preDefinedModel)

		#self.y_valid = np.array(valid_set[1], dtype=theano.config.floatX)

	def EVOEvaluate(self, individual):

		self.model.updateParameters(np.asarray(individual))
		return self.model.updateOutput(inputData=self.x_train, targets=self.y_train)

	def testModel(self, individual):

		self.model.updateParameters(np.asarray(individual))
		return self.model.updateOutput(inputData=self.x_valid, targets=self.y_valid)

	def modelFit(self):
		raise NotImplementedError('Optimizers must override modelFit()')

class NNEVO(object):

	def __init__(self, preModel, lossFunction):

		self.EVOModel = preModel
		self.lossFunction = lossFunction

	def updateParameters(self, parameters):

		parametersCopy = parameters

		layersList = self.EVOModel.layers

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
					newParameters = parametersCopy[0:numberOfParameters]

				paramsToUpdate.append(newParameters)

				parametersCopy = np.delete(parametersCopy, range(numberOfParameters))

			layerToUpdate.set_weights(paramsToUpdate)

	def updateOutput(self, inputData, targets):
			
		self.output = self.EVOModel.predict(inputData);

		self.loss = calculateLoss(targets, self.output, lossFunction=self.lossFunction)

		return [self.loss,];

class DEOptimizer(Optimizer):

	def mutDE(self, y, a, b, c, f):
		size = len(y)
		for i in range(len(y)):
			y[i] = a[i] + f*(b[i]-c[i])
		return y

	def cxBinomial(self, x, y, cr):
		size = len(x)
		index = random.randrange(size)
		for i in range(size):
			if i == index or random.random() < cr:
			    x[i] = y[i]
		return x

	def cxExponential(self, x, y, cr):
		size = len(x)
		index = random.randrange(size)
		# Loop on the indices index -> end, then on 0 -> index
		for i in chain(range(index, size), range(0, index)):
			x[i] = y[i]
			if random.random() < cr:
			    break
		return x

	def computeValidationFitness(self, hallOfFame):
	
		bestFitness = float('inf')

		for ind in hallOfFame:
			fitness = self.testModel(ind)
			if fitness < bestFitness:
				bestFitness = fitness
		return fitness

	def modelFit(self):

		print('Start of evolution...')

		NDIM = self.totalNumberOfParameters

		creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
#		creator.create("Individual", array.array, typecode='f', fitness=creator.FitnessMin)
		creator.create("Individual", list, fitness=creator.FitnessMin)

		toolbox = base.Toolbox()
		toolbox.register("attr_float", random.uniform, -10, 10)
		toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, NDIM)
		toolbox.register("population", tools.initRepeat, list, toolbox.individual)
		toolbox.register("mutate", self.mutDE, f=0.8)
		toolbox.register("mate", self.cxExponential, cr=0.8)
		toolbox.register("select", tools.selRandom, k=3)
		toolbox.register("evaluate", self.EVOEvaluate)

		MU = self.PopulationSize
		NGEN = self.numberOfEpochs  

		runMax = 300
		patience = 100
		epoch=0
		run=0
		found = False
		CPNamePOP = None

		#Look for checkpoint populations

		for i in range(runMax,-1,-1):
			for j in range(self.numberOfEpochs,-1,-1):	
				epochStr = str(j)
				runStr = str(i)

				if (os.path.exists('/scratch/nwv-632-aa/CP/pop-'+runStr+'-'+epochStr+'.p')):
					CPNamePOP = '/scratch/nwv-632-aa/CP/pop-'+runStr+'-'+epochStr+'.p'
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

		if (os.path.exists('fitness.p') and found):
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

			currentBestValidationFitness = self.computeValidationFitness(hof)

			if currentBestValidationFitness < lastBestValidationFitness:
				iterationsWithoutImprovement = 0
			else:
				iterationsWithoutImprovement += 1

			bestTrainFitnessHist.append(hof[0].fitness.values[0])
			bestValidFitnessHist.append(currentBestValidationFitness)

			pickle.dump(bestTrainFitnessHist, open('train_fitness.p', 'wb'))
			pickle.dump(bestValidFitnessHist, open('valid_fitness.p', 'wb'))
			pickle.dump(pop, open('/scratch/nwv-632-aa/CP/pop-'+str(run)+'-'+str(g)+'.p', 'wb'))

			g+=1

			print(logbook.stream)

		print("Best fitness found is:", hof[0].fitness.values[0])

		self.bestIndividuals = hof
		pickle.dump(logbook, open('DEOptimizer_logbook.p', 'wb'))
		buildAndSaveModels(self)
		return logbook

class SGDOptimizer(Optimizer):
	
	def modelFit(self):
		earlyStopping = EarlyStopping(monitor='val_loss', patience=30)
		hist = self.model.EVOModel.fit(self.x_train, self.y_train, batch_size=self.PopulationSize, epochs=self.numberOfEpochs, verbose=0, validation_data=(self.x_valid, self.y_valid), callbacks=[earlyStopping])
		self.model.EVOModel.save('SGDTrained.h5')
		pickle.dump(hist.history, open('SGDOptimizer_history.p', 'wb'))
		return hist

