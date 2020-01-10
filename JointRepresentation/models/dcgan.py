import tensorflow as tf
import numpy as np
from models.model_lstm import Model as Model_lstm
from config import Config


def define_eeg_encoder():
	model = tf.keras.models.load_model("models/latest.h5")
	encoder = tf.keras.Model(inputs=model.layers[0].input, outputs=model.layers[13].output)
	encoder.trainable = False
	return encoder

def define_discriminator(in_shape=(112,112,3), in_shape_eeg_features=512):
	# label input
	in_features = tf.keras.Input(shape=(in_shape_eeg_features,))

	# image input
	in_image = tf.keras.Input(shape=in_shape)

	# downsample
	fe = tf.keras.layers.Conv2D(128, (3,3), strides=(2,2), padding='same')(in_image)
	fe = tf.keras.layers.BatchNormalization()(fe)
	fe = tf.keras.layers.LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe = tf.keras.layers.Conv2D(128, (4,4), strides=(2,2), padding='same')(fe)
	fe = tf.keras.layers.BatchNormalization()(fe)
	fe = tf.keras.layers.LeakyReLU(alpha=0.2)(fe)

	# downsample
	fe = tf.keras.layers.Conv2D(128, (4,4), strides=(2,2), padding='same')(fe)
	fe = tf.keras.layers.BatchNormalization()(fe)
	fe = tf.keras.layers.LeakyReLU(alpha=0.2)(fe)

	# flatten feature maps
	fe = tf.keras.layers.Flatten()(fe)
	# dropout
	fe = tf.keras.layers.Dropout(0.2)(fe)

	merge = tf.keras.layers.Concatenate()([fe, in_features])

	out_layer = tf.keras.layers.Dense(512, activation='sigmoid')(fe)
	out_layer = tf.keras.layers.Dropout(0.2)(out_layer)

	out_layer = tf.keras.layers.Dense(256, activation='sigmoid')(fe)
	out_layer = tf.keras.layers.Dropout(0.2)(out_layer)

	# output
	out_layer = tf.keras.layers.Dense(1, activation='sigmoid')(fe)
	# define model
	model = tf.keras.Model([in_image, in_features], out_layer)
	return model


def define_generator(latent_dim=100, in_shape_eeg_features=512):
	in_features = tf.keras.Input(shape=(in_shape_eeg_features,))
	# image generator input
	in_lat = tf.keras.Input(shape=(latent_dim,))

	merge = tf.keras.layers.Concatenate()([in_features, in_lat])

	# foundation for 7x7 image
	n_nodes = 256 * 14 * 14
	gen = tf.keras.layers.Dense(n_nodes)(merge)
	gen = tf.keras.layers.LeakyReLU(alpha=0.2)(gen)
	gen = tf.keras.layers.Reshape((14, 14, 256))(gen)

	# upsample to 14x14
	gen = tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
	gen = tf.keras.layers.BatchNormalization()(gen)
	gen = tf.keras.layers.LeakyReLU(alpha=0.2)(gen)
	
	# upsample to 28x28
	gen = tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
	gen = tf.keras.layers.BatchNormalization()(gen)
	gen = tf.keras.layers.LeakyReLU(alpha=0.2)(gen)
	
	# upsample to 56x56
	gen = tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
	gen = tf.keras.layers.BatchNormalization()(gen)
	gen = tf.keras.layers.LeakyReLU(alpha=0.2)(gen)

	gen = tf.keras.layers.Conv2D(128, (7,7), padding='same')(gen)
	gen = tf.keras.layers.LeakyReLU()(gen)

	# output
	out_layer = tf.keras.layers.Conv2D(3, (5,5), activation='sigmoid', padding='same')(gen)
	# define model
	model = tf.keras.Model([in_lat, in_features], out_layer)
	return model
