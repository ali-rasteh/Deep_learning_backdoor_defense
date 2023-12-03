import keras
import tensorflow as tf
import keras.backend as K
from keras import initializers
import numpy as np


def Net():
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
	def __init__(self, channel_indices, **kwargs):
		super().__init__(**kwargs)
		self.channel_indices = channel_indices

	def call(self, inputs):
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
		sublayer_config = config.pop("channel_indices")
		channel_indices = keras.saving.deserialize_keras_object(sublayer_config)
		return cls(channel_indices, **config)


@keras.saving.register_keras_serializable(package="MyLayers")
class CompareAndSelectLayer(keras.layers.Layer):
	def __init__(self, **kwargs):
		super(CompareAndSelectLayer, self).__init__(**kwargs)

	def call(self, inputs):
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
