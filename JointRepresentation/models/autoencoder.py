import tensorflow as tf
import numpy as np
from models.model_lstm import Model as Model_lstm
from config import Config


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
	def __init__(self, intermediate_dim, original_dim, dataset):
		super(Autoencoder, self).__init__()

		# model = Model_lstm()
		# init_model(model, dataset)
		model = tf.keras.models.load_model("models/latest.h5")
		
		# model.summary()
		# tf.keras.utils.plot_model(model, to_file='model.png')

		self.encoder = tf.keras.Model(inputs=model.layers[0].input, outputs=model.layers[13].output)
		self.encoder.trainable = False
		# self.decoder = Decoder(intermediate_dim=intermediate_dim, original_dim=original_dim)
		self.decoder = define_generator(intermediate_dim, original_dim)

	def call(self, input_features):
		code = self.encoder(input_features)
		reconstructed = self.decoder(code)
		return reconstructed

def define_generator(latent_dim, original_dim):
	# image generator input
	in_lat = tf.keras.Input(shape=(latent_dim,))
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	gen = tf.keras.layers.Dense(n_nodes, activation="relu")(in_lat)
	gen = tf.keras.layers.Reshape((7, 7, 128))(gen)

	# upsample to 14x14
	gen = tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', activation="relu")(gen)
	# upsample to 28x28
	gen = tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', activation="relu")(gen)
	# upsample to 56x56
	gen = tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', activation="relu")(gen)

	# output
	out_layer = tf.keras.layers.Conv2D(3, (7,7), activation='tanh', padding='same')(gen)
	# out_layer = Dense(original_dim)(gen)
	# define model
	model = tf.keras.Model(inputs=in_lat, outputs=out_layer)
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
	gen = Conv2DTranspose(1, (4,4), strides=(2,2), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	# output
	out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
	# define model
	model = Model([in_lat, in_label], out_layer)
	return model


def init_model(model, dataset):
	model.compile(
		optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']
	)

	sample_dataset = dataset.take(1)
	# model.fit(sample_dataset)
	for (batch, (record_sample, img_tensor)) in enumerate(sample_dataset):
		model(record_sample)
	
	return model