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

		self.latent_encoder = tf.keras.Model(inputs=model.layers[0].input, outputs=model.layers[13].output)
		self.latent_encoder.trainable = False

		self.encoder = tf.keras.Sequential([
			self.latent_encoder,
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(intermediate_dim + intermediate_dim)
		])

		# self.decoder = Decoder(intermediate_dim=intermediate_dim, original_dim=original_dim)
		self.decoder = define_generator(intermediate_dim, original_dim)

	@tf.function
	def sample(self, eps=None):
		if eps is None:
			eps = tf.random.normal(shape=(100, self.latent_dim))
		return self.decode(eps, apply_sigmoid=True)

	def encode(self, x):
		mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
		return mean, logvar

	def reparameterize(self, mean, logvar):
		eps = tf.random.normal(shape=mean.shape)
		return eps * tf.exp(logvar * .5) + mean

	def decode(self, z, apply_sigmoid=False):
		logits = self.decoder(z)
		if apply_sigmoid:
			probs = tf.sigmoid(logits)
			return probs

		return logits

	def call(self, x):
		pass



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