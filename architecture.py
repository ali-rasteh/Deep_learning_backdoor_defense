import keras
import tensorflow as tf
import keras.backend as K
from keras import initializers
import numpy as np


def Net():
	"""
	Constructs a Convolutional Neural Network (CNN) model using Keras.
	The network architecture consists of:
	- An input layer with shape (55, 47, 3).
	- Three convolutional layers with ReLU activation and max pooling layers for feature extraction.
	- Two interpretation models:
		- The first interpretation model flattens the output of the third pooling layer and applies a dense layer.
		- The second interpretation model applies an additional convolutional layer, flattens it, and applies a dense layer.
	- The outputs of the two interpretation models are merged and passed through a ReLU activation layer.
	- A dropout layer with a dropout rate of 0.5.
	- An output dense layer with 1283 units and softmax activation.
	Returns:
		keras.Model: The constructed Keras model.
	"""

	# define input
	x = keras.Input(shape=(55, 47, 3), name='input')
	# feature extraction
	conv_1 = keras.layers.Conv2D(20, (4, 4), activation='relu', name='conv_1')(x)
	pool_1 = keras.layers.MaxPooling2D((2, 2), name='pool_1')(conv_1)
	conv_2 = keras.layers.Conv2D(40, (3, 3), activation='relu', name='conv_2')(pool_1)
	pool_2 = keras.layers.MaxPooling2D((2, 2), name='pool_2')(conv_2)
	conv_3 = keras.layers.Conv2D(60, (3, 3), activation='relu', name='conv_3')(pool_2)
	pool_3 = keras.layers.MaxPooling2D((2, 2), name='pool_3')(conv_3)
	# first interpretation model
	flat_1 = keras.layers.Flatten()(pool_3)	
	fc_1 = keras.layers.Dense(160, name='fc_1')(flat_1)
	# second interpretation model
	conv_4 = keras.layers.Conv2D(80, (2, 2), activation='relu', name='conv_4')(pool_3)
	flat_2 = keras.layers.Flatten()(conv_4)
	fc_2 = keras.layers.Dense(160, name='fc_2')(flat_2)
	# merge interpretation
	merge = keras.layers.Add()([fc_1, fc_2])
	add_1 = keras.layers.Activation('relu')(merge)
	drop = keras.layers.Dropout(0.5)
	# output
	y_hat = keras.layers.Dense(1283, activation='softmax', name='output')(add_1)
	model = keras.Model(inputs=x, outputs=y_hat)
	# summarize layers
	#print(model.summary())
	# plot graph
	#plot_model(model, to_file='model_architecture.png')

	return model


def Net_pruned(prune_channel_indices=None):
	"""
	Builds a pruned neural network model using Keras.
	Parameters:
	prune_channel_indices (list or None): List of channel indices to prune. If None, no pruning is applied.
	Returns:
	keras.Model: A Keras Model instance representing the pruned neural network.
	The model architecture includes:
	- Input layer with shape (55, 47, 3)
	- Three convolutional layers with ReLU activation and max pooling
	- A custom channel pruning layer applied after the third pooling layer
	- Two interpretation models:
		- First interpretation model: Flatten -> Dense layer
		- Second interpretation model: Convolutional layer -> Flatten -> Dense layer
	- Merging of the two interpretation models using addition and ReLU activation
	- Dropout layer with a rate of 0.5
	- Output layer with 1283 units and softmax activation
	"""

	# define input
	x = keras.Input(shape=(55, 47, 3), name='input')
	# feature extraction
	conv_1 = keras.layers.Conv2D(20, (4, 4), activation='relu', name='conv_1')(x)
	pool_1 = keras.layers.MaxPooling2D((2, 2), name='pool_1')(conv_1)
	conv_2 = keras.layers.Conv2D(40, (3, 3), activation='relu', name='conv_2')(pool_1)
	pool_2 = keras.layers.MaxPooling2D((2, 2), name='pool_2')(conv_2)
	conv_3 = keras.layers.Conv2D(60, (3, 3), activation='relu', name='conv_3')(pool_2)
	pool_3 = keras.layers.MaxPooling2D((2, 2), name='pool_3')(conv_3)
	pruned_pool_3 = ChannelPruningLayer(prune_channel_indices, name='pruned_pool_3')(pool_3)
	# first interpretation model
	flatten_1 = keras.layers.Flatten()(pruned_pool_3)	
	fc_1 = keras.layers.Dense(160, name='fc_1')(flatten_1)
	# second interpretation model
	conv_4 = keras.layers.Conv2D(80, (2, 2), activation='relu', name='conv_4')(pruned_pool_3)
	flatten_2 = keras.layers.Flatten()(conv_4)
	fc_2 = keras.layers.Dense(160, name='fc_2')(flatten_2)
	# merge interpretation
	merge = keras.layers.Add()([fc_1, fc_2])
	add_1 = keras.layers.Activation('relu')(merge)
	drop = keras.layers.Dropout(0.5)
	# output
	y_hat = keras.layers.Dense(1283, activation='softmax', name='output')(add_1)
	model = keras.Model(inputs=x, outputs=y_hat)
	# summarize layers
	#print(model.summary())
	# plot graph
	#plot_model(model, to_file='model_architecture.png')

	return model


def Good_Net(net, net_pruned):
	"""
	Constructs a Keras model that takes an input tensor and processes it through two given networks,
	then compares and selects the output using a custom comparison layer.
	Args:
		net (keras.Model): The first neural network model to process the input.
		net_pruned (keras.Model): The second neural network model to process the input, typically a pruned version of the first model.
	Returns:
		keras.Model: A Keras model that takes an input tensor, processes it through both provided networks,
					 and outputs the result of a custom comparison layer applied to the outputs of the two networks.
	"""

	x = keras.Input(shape=(55, 47, 3), name='input')

	# net_model = Net()
	# net_pruned_model = Net_pruned(prune_channel_indices)
	net_model = net
	net_pruned_model = net_pruned

	net_output = net_model(x)
	net_pruned_output = net_pruned_model(x)

	comparison_layer = CompareAndSelectLayer()([net_output, net_pruned_output])
	Good_model = keras.Model(inputs=x, outputs=comparison_layer)

	return Good_model


@keras.saving.register_keras_serializable(package="MyLayers")
class ChannelPruningLayer(keras.layers.Layer):
	"""
	A custom Keras layer for channel pruning in a neural network.
	This layer allows for the pruning (zeroing out) of specific channels in the input tensor.
	It takes a list of channel indices to be pruned and sets the corresponding channels to zero.
	Attributes:
		channel_indices (list): A list of indices of the channels to be pruned.
	Methods:
		call(inputs):
			Applies the channel pruning to the input tensor.
		get_config():
			Returns the configuration of the layer for serialization.
		from_config(config):
			Instantiates the layer from a configuration dictionary.
	"""

	def __init__(self, channel_indices, **kwargs):
		super().__init__(**kwargs)
		self.channel_indices = channel_indices

	def call(self, inputs):
		"""
		Applies a mask to the input tensor by zeroing out specified channels.
		Args:
			inputs (tf.Tensor): The input tensor of shape (batch_size, height, width, channels).
		Returns:
			tf.Tensor: The output tensor with specified channels zeroed out.
		Notes:
			- If `self.channel_indices` is not None, it should be a list of channel indices to be zeroed out.
			- For each channel index in `self.channel_indices`, the corresponding channel in the input tensor is replaced with zeros.
			- The output tensor is obtained by element-wise multiplication of the input tensor and a mask tensor.
		"""

		output = inputs
		if self.channel_indices is not None:
			for channel_index in self.channel_indices:
				zeros = tf.zeros_like(inputs[:, :, :, channel_index:channel_index + 1])
				# output = tf.concat([output[:, :, :, :channel_index], zeros, output[:, :, :, channel_index+1:]], axis=-1)
				mask = tf.ones_like(inputs)
				mask = tf.concat([mask[:, :, :, :channel_index], zeros, mask[:, :, :, channel_index+1:]], axis=-1)
				output = tf.math.multiply(output, mask)
		return output
	
	def get_config(self):
		"""
		Returns the configuration of the layer.
		This method retrieves the base configuration from the parent class and 
		updates it with the configuration specific to this layer, including the 
		serialized `channel_indices`.
		Returns:
			dict: A dictionary containing the configuration of the layer.
		"""

		base_config = super().get_config()
		# config.update({
		# 	"channel_indices": self.channel_indices,
		# })
		# return config
		config = {
            "channel_indices": keras.saving.serialize_keras_object(self.channel_indices),
        }
		return {**base_config, **config}
	
	@classmethod
	def from_config(cls, config):
		"""
		Creates an instance of the class from a configuration dictionary.
		Args:
			config (dict): A dictionary containing the configuration for the instance.
				It should include a key "channel_indices" which will be used to 
				deserialize the corresponding sublayer configuration.
		Returns:
			An instance of the class initialized with the provided configuration.
		Raises:
			KeyError: If the "channel_indices" key is not found in the config dictionary.
		"""

		sublayer_config = config.pop("channel_indices")
		channel_indices = keras.saving.deserialize_keras_object(sublayer_config)
		return cls(channel_indices, **config)


@keras.saving.register_keras_serializable(package="MyLayers")
class CompareAndSelectLayer(keras.layers.Layer):
	"""
	A custom Keras layer that compares the indices of the maximum elements 
	from two input tensors and selects the output based on the comparison.
	If the indices of the maximum elements in the two input tensors are equal, 
	the original output is returned. Otherwise, a tensor of zeros is returned. 
	Additionally, a flag indicating whether the indices were different (1 if different, 
	0 if the same) is appended to the output.
	Methods
	-------
	__init__(**kwargs)
		Initializes the CompareAndSelectLayer.
	call(inputs)
		Compares the indices of the maximum elements from the input tensors and 
		returns the selected output with an appended flag.
	Parameters
	----------
	inputs : list of tensors
		A list containing two tensors: `net_output` and `net_pruned_output`.
	Returns
	-------
	tensor
		A tensor with the selected output and an appended flag indicating 
		whether the indices of the maximum elements were different.
	"""

	def __init__(self, **kwargs):
		super(CompareAndSelectLayer, self).__init__(**kwargs)

	def call(self, inputs):
		"""
		Processes the input tensors and returns a combined output with an indicator flag.
		Args:
			inputs (tuple): A tuple containing two tensors:
				- net_output (tf.Tensor): The output tensor from the original network.
				- net_pruned_output (tf.Tensor): The output tensor from the pruned network.
		Returns:
			tf.Tensor: A tensor containing the selected output from the original network 
			and an additional flag indicating whether the indices of the maximum elements 
			in the original and pruned network outputs were different (1 if different, 0 if the same).
		"""

		net_output, net_pruned_output = inputs

		# Get the indices of the maximum elements
		max_index_net = tf.argmax(net_output, axis=-1)
		max_index_net_pruned = tf.argmax(net_pruned_output, axis=-1)

		# Check if the indices of the maximum elements are equal
		indices_equal = tf.equal(max_index_net, max_index_net_pruned)
		indices_equal = tf.expand_dims(indices_equal, -1)
		indices_equal_inv_float = 1-tf.cast(indices_equal, tf.float32)

		# If indices are equal, return the original output, otherwise return zeros
		output_with_zeros = tf.zeros_like(net_output)
		selected_output = tf.where(tf.broadcast_to(indices_equal, tf.shape(net_output)), net_output, output_with_zeros)

		# Append a flag indicating whether the indices were equal (1 if different, 0 if the same)
		combined_output_with_indicator = tf.concat([selected_output, indices_equal_inv_float], axis=-1)

		return combined_output_with_indicator
	

K.clear_session()
model = Net()
