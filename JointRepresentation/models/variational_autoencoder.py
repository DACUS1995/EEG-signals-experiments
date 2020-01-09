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
			units=original_dim
		)

		self.reshape_layer = tf.keras.layers.Reshape((56, 56, 3,))

	def call(self, code):
		activation = self.hidden_layer(code)
		out = self.output_layer(activation)
		return self.reshape_layer(out)


class Autoencoder(tf.keras.Model):
	def __init__(self, intermediate_dim, original_dim, dataset):
		super(Autoencoder, self).__init__()

		model = tf.keras.models.load_model("models/latest.h5")
		
		latent_encoder = tf.keras.Model(inputs=model.layers[0].input, outputs=model.layers[13].output)
		latent_encoder.trainable = False

		latent_encoder_input = latent_encoder.layers[0].input
		conditioning_input = tf.keras.Input(shape=(2,), name="Input_2")

		out_latent_encoder = latent_encoder(latent_encoder_input)

		out = tf.keras.layers.Flatten()(out_latent_encoder)
		out = tf.keras.layers.Concatenate(axis=1)([out, conditioning_input])
		out = tf.keras.layers.Dense(intermediate_dim + intermediate_dim)(out)

		self.encoder = tf.keras.Model(inputs=[latent_encoder_input, conditioning_input], outputs=out)

		# self.encoder = tf.keras.Sequential([
		# 	self.latent_encoder,
		# 	tf.keras.layers.Flatten(),
		# 	tf.keras.layers.Dense(intermediate_dim + intermediate_dim)
		# ])


		# self.decoder = Decoder(intermediate_dim=intermediate_dim + 2, original_dim=original_dim)
		self.decoder = define_generator(intermediate_dim + 2, original_dim)

	@tf.function
	def sample(self, eps=None, cond=[1, 0]):
		if eps is None:
			eps = tf.random.normal(shape=(100, self.latent_dim))
		return self.decode(eps, apply_sigmoid=True)

	def encode(self, x, y):
		mean, logvar = tf.split(self.encoder([x, y]), num_or_size_splits=2, axis=1)
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

	def call(self, record_sample, label):
		record_sample = tf.cast(record_sample, tf.float32)
		label = tf.cast(label, tf.float32)

		mean, logvar = self.encode(record_sample, label)
		z = self.reparameterize(mean, logvar)
		z = tf.concat([z, label], axis=1)
		probs = self.decode(z, apply_sigmoid=True)
		return probs



def define_generator(latent_dim, original_dim):
	# image generator input
	in_lat = tf.keras.Input(shape=(latent_dim,))
	
	gen = tf.keras.layers.Dense(784, activation="relu")(in_lat)

	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	gen = tf.keras.layers.Dense(n_nodes, activation="relu")(gen)
	gen = tf.keras.layers.Reshape((7, 7, 128))(gen)

	# upsample to 14x14
	gen = tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', activation="relu")(gen)
	gen = tf.keras.layers.BatchNormalization()(gen)
	# upsample to 28x28
	gen = tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', activation="relu")(gen)
	gen = tf.keras.layers.BatchNormalization()(gen)

	# upsample to 56x56
	gen = tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', activation="relu")(gen)
	gen = tf.keras.layers.BatchNormalization()(gen)

	# output
	out_layer = tf.keras.layers.Conv2D(3, (3,3), padding='same')(gen)
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