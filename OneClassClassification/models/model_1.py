import tensorflow as tf
import numpy as np

from config import Config

model = tf.keras.models.Sequential([
	# tf.keras.layers.Flatten(input_shape=(Config.RECORDING_NUM_SAMPLES, len(Config.SENSORS_LABELS))),
	tf.keras.layers.Reshape((Config.RECORDING_NUM_SAMPLES, len(Config.SENSORS_LABELS), 1), input_shape=(Config.RECORDING_NUM_SAMPLES, len(Config.SENSORS_LABELS))),
	tf.keras.layers.Conv2D(32, 3, activation='relu'),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.MaxPool2D(),

	tf.keras.layers.Conv2D(32, 3, activation='relu'),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.MaxPool2D(),

	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(512, activation='relu'),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Dense(2, activation='softmax')
])

# model = tf.keras.models.Sequential([
# 	tf.keras.layers.Flatten(input_shape=(28, 28)),
# 	tf.keras.layers.Dense(128, activation='relu'),
# 	tf.keras.layers.Dropout(0.2),
# 	tf.keras.layers.Dense(10, activation='softmax')
# ])

# class MyModel(tf.keras.Model):
#   def __init__(self):
# 	super().__init__()
# 		self.conv1 = Conv2D(32, 3, activation='relu')
# 		self.flatten = Flatten()
# 		self.d1 = Dense(128, activation='relu')
# 		self.d2 = Dense(10, activation='softmax')

# 	def call(self, x):
# 		x = self.conv1(x)
# 		x = self.flatten(x)
# 		x = self.d1(x)
# 		return self.d2(x)