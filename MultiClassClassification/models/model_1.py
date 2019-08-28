import tensorflow as tf
import numpy as np

from config import Config

class Model(tf.keras.Model):
	def __init__(self):
		super().__init__()
		self.reshape = tf.keras.layers.Reshape((Config.RECORDING_NUM_SAMPLES, len(Config.SENSORS_LABELS), 1), input_shape=(Config.RECORDING_NUM_SAMPLES, len(Config.SENSORS_LABELS)))
		self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding="same")
		self.pool = tf.keras.layers.MaxPool2D(pool_size=(2,2), padding="same")
		self.norm1 = tf.keras.layers.BatchNormalization()
		self.drop1 = tf.keras.layers.Dropout(0.5)


		self.conv2 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding="same")
		self.norm2 = tf.keras.layers.BatchNormalization()
		self.drop2 = tf.keras.layers.Dropout(0.5)


		self.conv3 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding="same")
		self.norm3 = tf.keras.layers.BatchNormalization()
		self.drop3 = tf.keras.layers.Dropout(0.5)


		self.conv4 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding="same")
		self.norm4 = tf.keras.layers.BatchNormalization()
		self.drop4 = tf.keras.layers.Dropout(0.5)


		self.conv5 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding="same")
		self.norm5 = tf.keras.layers.BatchNormalization()
		self.drop5 = tf.keras.layers.Dropout(0.5)


		self.flatten = tf.keras.layers.Flatten()
		self.d1 = tf.keras.layers.Dense(256, activation='relu')
		self.dropout = tf.keras.layers.Dropout(0.2)
		self.d2 = tf.keras.layers.Dense(2, activation='softmax')

	def call(self, x, training=False):
		# out = self.reshape(x)
		# out = tf.expand_dims(x, 2)
		out = tf.reshape(x, [-1, Config.RECORDING_NUM_SAMPLES, 14, 1])

		out = self.conv1(out)
		out = self.pool(out)
		if training:
			out = self.drop1(out)
			out = self.norm1(out)

		out = self.conv4(out)

		if training:
			out = self.drop4(out)
			out = self.norm4(out)

		# out = self.conv3(out)
		# out = self.norm3(out)

		# out = self.conv4(out)
		# out = self.norm4(out)

		if training:
			out = self.dropout(out)

		out = self.flatten(out)
		# out = self.d1(out)

		out = self.d2(out)
		return out

model = Model()