from deap import base
from deap import creator
from deap import tools

import numpy as np

from itertools import chain

import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms

import random
import array
import pickle
import os.path

from Utils import countParameters, calculateLoss, lossFuncException, tensorElementsCount, buildAndSaveModels, find_last_improvement, batch_generator

class Optimizer(object):

	def __init__(self, x_train, y_train, x_valid, y_valid, preDefinedModel, n_epochs=100, popSize = 300, loss_function = 'mse'):

		self.numberOfEpochs = n_epochs
		self.PopulationSize = popSize
		self.x_train = x_train
		self.y_train = y_train
		self.x_valid = x_valid
		self.y_valid = y_valid
		self.loss_function = loss_function

		self.model = preDefinedModel

		self.totalNumberOfParameters = countParameters(preDefinedModel)

	def Evaluate(self, individual):

		self.updateParameters(np.asarray(individual, dtype='float32'))
		return self.updateOutput(inputData=self.x_train, targets=self.y_train)

	def testModel(self, individual):

		self.updateParameters(np.asarray(individual, dtype='float32'))
		return self.updateOutput(inputData=self.x_valid, targets=self.y_valid)

	def updateParameters(self, parameters):

		paramsCopy = torch.from_numpy(parameters)

		for param in self.model.parameters():
			numPar = tensorElementsCount(param)
			parSize = param.size()
			paramsubset = paramsCopy[0:numPar]
			param_size = param.size()
			param.data = paramsubset.view(param_size)
			try:
				paramsCopy = paramsCopy[numPar:]
			except ValueError:
				break
		self.model.cuda()

	def updateOutput(self, inputData, targets):
		self.model.eval()
		total_loss = 0
		data_loader = batch_generator(inputData, targets)
		for data, target in data_loader:
			if torch.cuda.is_available():
				data, target = data.cuda(), target.cuda()
			data, target = Variable(data, volatile=True), Variable(target)
			output = self.model.forward(data)
			loss = calculateLoss(output, target, lossFunction=self.loss_function)
			total_loss += loss.data[0]*data.size()[0]
			
		self.output = output

		self.loss = total_loss/inputData.size()[0]

		return [self.loss,]

	def modelFit(self):
		raise NotImplementedError('Optimizers must override modelFit()')

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
			if fitness[0] < bestFitness:
				bestFitness = fitness[0]
		return bestFitness

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
		toolbox.register("evaluate", self.Evaluate)

		MU = self.PopulationSize
		NGEN = self.numberOfEpochs  

		runMax = 300
		patience = 30
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

		if (os.path.exists('train_fitness.p') and os.path.exists('valid_fitness.p') and found):
			bestTrainFitnessHist = pickle.load(open('train_fitness.p', 'rb'))
			bestValidFitnessHist = pickle.load(open('valid_fitness.p', 'rb'))
			iterationsWithoutImprovement = find_last_improvement(bestValidFitnessHist)
			lastBestValidationFitness = bestValidFitnessHist[-1]
		else:
			bestTrainFitnessHist = []
			bestValidFitnessHist = []
			iterationsWithoutImprovement = 0
			lastBestValidationFitness = float('inf')

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

		currentBestValidationFitness = 0

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

			if currentBestValidationFitness <= lastBestValidationFitness:
				lastBestValidationFitness = currentBestValidationFitness
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

	def train(self, epoch):
		self.model.train()
		train_loader = batch_generator(self.x_train, self.y_train, self.PopulationSize)
		for batch_idx, (data, target) in enumerate(train_loader):
			if torch.cuda.is_available():
				data, target = data.cuda(), target.cuda()
			data, target = Variable(data), Variable(target)
			self.optim.zero_grad()
			output = self.model.forward(data)
			loss = calculateLoss(output, target, lossFunction=self.loss_function)
			loss.backward()
			self.optim.step()
			if batch_idx % 10 == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), self.x_train.size()[0], 100. * batch_idx * len(data) / self.x_train.size()[0], loss.data[0]))

	def test(self):
		self.model.eval()
		test_loss = 0
		test_loader = batch_generator(self.x_valid, self.y_valid, self.PopulationSize)
		for data, target in test_loader:
			if torch.cuda.is_available():
				data, target = data.cuda(), target.cuda()
			data, target = Variable(data, volatile=True), Variable(target)
			output = self.model.forward(data)
			loss = calculateLoss(output, target, lossFunction=self.loss_function)
			test_loss += loss.data[0]*data.size()[0]

		test_loss/=self.x_valid.size()[0]
		print('Test Loss: {}'.format(test_loss))
		return test_loss

	def test__(self):
		self.updateParameters(np.random.random(self.totalNumberOfParameters))
		loss = self.updateOutput(self.x_valid, self.y_valid)
		print('Test Loss: {}'.format(loss[0]))
		return loss[0]
	
	def modelFit(self):

		self.optim = optim.Adam(self.model.parameters())
		patience = 30
		epoch = 1
		lastBestValidationLoss = float('inf')
		iterationsWithoutImprovement = 0

		while ((epoch <= self.numberOfEpochs) and (iterationsWithoutImprovement < patience)):
			self.train(epoch)
			epoch+=1
			currentValidationLoss = self.test()
		
			if currentValidationLoss<lastBestValidationLoss:
				iterationsWithoutImprovement = 0
				lastBestValidationLoss = currentValidationLoss
			else:
				iterationsWithoutImprovement+=1
			
		pickle.dump(self.model, open('SGDTrained.p', 'wb'))

