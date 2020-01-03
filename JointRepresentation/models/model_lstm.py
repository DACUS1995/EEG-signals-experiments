import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

from config import Config

def create_model():
	inputs = tf.keras.Input(shape=(Config.RECORDING_NUM_SAMPLES, len(Config.SENSORS_LABELS)))
	out = tf.keras.layers.Reshape((Config.RECORDING_NUM_SAMPLES, len(Config.SENSORS_LABELS), 1))(inputs)

	out = tf.keras.layers.Conv2D(32, 3, activation='relu', padding="same")(out)
	out = tf.keras.layers.MaxPool2D(pool_size=(2,2), padding="same")(out)
	out = tf.keras.layers.Dropout(0.5)(out)
	out = tf.keras.layers.BatchNormalization()(out)

	out = tf.keras.layers.Conv2D(32, 3, activation='relu', padding="same")(out)
	out = tf.keras.layers.Dropout(0.5)(out)
	out = tf.keras.layers.BatchNormalization()(out)
	
	out = tf.keras.layers.Reshape((95, 32 * 7))(out)
	out = tf.keras.layers.LSTM(126, return_sequences = True)(out) #return_sequences = True for stacking
	out = tf.keras.layers.LSTM(126)(out)

	out = tf.keras.layers.Flatten()(out)
	out = tf.keras.layers.Dense(512, activation='relu')(out)
	out = tf.keras.layers.Dropout(0.2)(out)
	out = tf.keras.layers.Dense(2, activation='softmax')(out)

	autoencoder = keras.Model(inputs, out, name='autoencoder')
	autoencoder.summary()
	return autoencoder

class Model(tf.keras.Model):
	def __init__(self):
		super().__init__()
		self.reshape = tf.keras.layers.Reshape((Config.RECORDING_NUM_SAMPLES, len(Config.SENSORS_LABELS)), input_shape=(Config.RECORDING_NUM_SAMPLES, len(Config.SENSORS_LABELS)))
		
		self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding="same")
		self.pool = tf.keras.layers.MaxPool2D(pool_size=(2,2), padding="same")
		self.norm1 = tf.keras.layers.BatchNormalization()
		self.drop1 = tf.keras.layers.Dropout(0.5)

		self.conv2 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding="same")
		self.norm2 = tf.keras.layers.BatchNormalization()
		self.drop2 = tf.keras.layers.Dropout(0.5)
		
		self.lstm1 = tf.keras.layers.LSTM(126, return_sequences = True) #return_sequences = True for stacking
		self.lstm2 = tf.keras.layers.LSTM(126)

		self.flatten = tf.keras.layers.Flatten()
		self.d1 = tf.keras.layers.Dense(512, activation='relu')
		self.dropout = tf.keras.layers.Dropout(0.2)
		self.d2 = tf.keras.layers.Dense(2, activation='softmax')

	def call(self, x, training=False):
		# out = self.reshape(x)
		# out = tf.expand_dims(x, 2)
		out = tf.reshape(x, [-1, Config.RECORDING_NUM_SAMPLES, len(Config.SENSORS_LABELS), 1])

		out = self.conv1(out)
		out = self.pool(out)
		out = self.drop1(out)
		if training:
			out = self.norm1(out)

		out = self.conv2(out)
		if training:
			out = self.drop2(out)
			out = self.norm2(out)

		out = tf.reshape(out, [-1, 100, 32 * 7])
		out = tf.cast(out, dtype=tf.float32)
		out = self.lstm1(out)
		out = self.lstm2(out)

		out = self.flatten(out)
		out = self.d1(out)
		if training:
			out = self.dropout(out)
		out = self.d2(out)
		return out

		

model = Model()