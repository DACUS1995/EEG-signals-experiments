import tensorflow as tf
import numpy as np
from models.model_lstm import Model as Model_lstm


class Decoder(tf.keras.layers.Layer):
	def __init__(self, intermediate_dim, original_dim):
		super(Decoder, self).__init__()
		self.hidden_layer = tf.keras.layers.Dense(
			units=intermediate_dim,
			activation=tf.nn.relu,
			kernel_initializer='he_uniform'
		)
		self.output_layer = tf.keras.layers.Dense(
			units=original_dim,
			activation=tf.nn.sigmoid
		)

	def call(self, code):
		activation = self.hidden_layer(code)
		return self.output_layer(activation)


class Autoencoder(tf.keras.Model):
	def __init__(self, intermediate_dim, original_dim):
		super(Autoencoder, self).__init__()

		model = Model_lstm()
		model.load_weights("models/model_lstm.h5")

		self.encoder = tf.keras.Model(inputs=model.input, outputs=model.layers[10].output)
		self.encoder.trainable = False
		self.decoder = Decoder(intermediate_dim=intermediate_dim, original_dim=original_dim)

	def call(self, input_features):
		code = self.encoder(input_features)
		reconstructed = self.decoder(code)
		return reconstructed

def define_generator(latent_dim=512):
	# linear multiplication
	n_nodes = 7 * 7
	# image generator input
	in_lat = Input(shape=(latent_dim,))
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	gen = Dense(n_nodes)(in_lat)
	gen = LeakyReLU(alpha=0.2)(gen)
	gen = Reshape((7, 7, 128))(gen)
	merge = gen
	# upsample to 14x14
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)
	gen = LeakyReLU(alpha=0.2)(gen)
	# upsample to 28x28
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	# output
	out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
	# define model
	model = Model([in_lat, in_label], out_layer)
	return model


def define_generator_old(latent_dim):
	# label input
	in_label = Input(shape=(512,))
	# linear multiplication
	n_nodes = 7 * 7
	li = Dense(n_nodes)(in_label)
	# reshape to additional channel
	li = Reshape((7, 7, 1))(li)
	# image generator input
	in_lat = Input(shape=(latent_dim,))
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	gen = Dense(n_nodes)(in_lat)
	gen = LeakyReLU(alpha=0.2)(gen)
	gen = Reshape((7, 7, 128))(gen)
	# merge image gen and label input
	merge = Concatenate()([gen, li])
	# upsample to 14x14
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)
	gen = LeakyReLU(alpha=0.2)(gen)
	# upsample to 28x28
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	# output
	out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
	# define model
	model = Model([in_lat, in_label], out_layer)
	return model