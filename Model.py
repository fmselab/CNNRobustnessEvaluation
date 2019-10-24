from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class Model:
	@staticmethod
	def build(width, height, depth, classes, finalAct="softmax"):
		# Initialize the model along with the input shape to be "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# If we are using "channels first", update the input shape and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# First convolutional layer of the net
		# The block has 32 filters, with 3x3 kernel and uses a RELU (Rectified Linear Unit)
		# activation function. We also apply batch normalization, max pooling and 25%
		# drop-out.
		# CONV => RELU => POOL
		model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape, strides=2))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(3,3)))
		model.add(Dropout(0.3))

		# Second convolutional layer of the net
		# (CONV => RELU) * 2 => POOL
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		# First (and only) set of FullyConnected => RELU layers
		# This layer is placed at the end of the network
		model.add(Flatten())
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		# Classifier, as the last layer of the net,as choosen by the user.
		# Use "sigmoid" for multi-label classification, or "softmax" for single-label
		# classification
		model.add(Dense(classes))
		model.add(Activation(finalAct))

		# return the constructed network architecture
		return model
