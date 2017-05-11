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
	model.add(Dense(64, init='uniform', input_dim=1978))
	model.add(Dropout(0.2))
	model.add(Activation('relu'))
	model.add(Dense(32, init="uniform"))
	model.add(Dropout(0.2))
	model.add(Activation('relu'))
	model.add(Dense(21, init="uniform"))
	model.add(Activation('sigmoid'))
	model.compile(loss='mean_squared_error', optimizer='sgd')
	return model
