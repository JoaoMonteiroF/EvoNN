from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
from keras import layers
from keras import models
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint


def MLP():
	model = Sequential()
	model.add(Dense(64, init='uniform', input_shape=1978))
	model.add(Dropout(0.2))
	model.add(Activation('relu'))
	model.add(Dense(32, init="uniform"))
	model.add(Dropout(0.2))
	model.add(Activation('relu'))
	model.add(Dense(21, init="uniform"))
	model.add(Activation('sigmoid'))
	model.compile(loss='mean_squared_error', optimizer='sgd')
	return model

def MLP_MNIST():
	model = Sequential()
	model.add(Flatten(input_shape=(28, 28, 1)))
	model.add(Dense(128))
	model.add(Dropout(0.5))
	model.add(Activation('sigmoid'))
	model.add(Dense(10))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	return model

def CNN_MNIST():
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
	model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(10, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	return model
