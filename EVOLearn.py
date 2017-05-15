import cPickle
import scipy.io

from keras.datasets import mnist
from keras.utils import to_categorical
from keras import backend as K

import numpy as np

import models_zoo
from Optimizer import Optimizer, DEOptimizer, NNEVO 
from Utils import buildAndSaveModels
from models_zoo import MLP_MNIST

############# Import data set

img_rows, img_cols = 28, 28
num_classes = 10

(x_train, y_train), (x_valid, y_valid) = mnist.load_data()

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

def main():

############# Create a Keras model and pass it to instantiate an optimizer

	numberOfEpochs = 1000
	popSize = 250

	model = MLP_MNIST()

	optimizer = DEOptimizer(x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid, preDefinedModel=model, n_epochs=numberOfEpochs, popSize = popSize, loss = 'mse')

	optimizer.modelFit()

	buildAndSaveModels(optimizer)

if __name__ == "__main__":
	main()
