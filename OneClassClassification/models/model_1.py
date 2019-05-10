import tensorflow as tf
import numpy as np

from config import Config

model = tf.keras.models.Sequential([
	# tf.keras.layers.Flatten(input_shape=(Config.RECORDING_NUM_SAMPLES, len(Config.SENSORS_LABELS))),
	tf.keras.layers.Reshape((Config.RECORDING_NUM_SAMPLES, len(Config.SENSORS_LABELS), 1), input_shape=(Config.RECORDING_NUM_SAMPLES, len(Config.SENSORS_LABELS))),
	tf.keras.layers.Conv2D(32, 3, activation='relu', padding="same"),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.MaxPool2D(pool_size=(2,2), padding="same"),

	tf.keras.layers.Conv2D(32, 3, activation='relu', padding="same"),
	tf.keras.layers.BatchNormalization(),
	# tf.keras.layers.MaxPool2D(pool_size=(2,2), padding="same"),

	tf.keras.layers.Conv2D(16, 6, activation='relu', padding="same"),
	tf.keras.layers.BatchNormalization(),
	# tf.keras.layers.MaxPool2D(pool_size=(2,2), padding="same"),

	tf.keras.layers.Conv2D(16, 3, activation='relu', padding="same"),
	tf.keras.layers.BatchNormalization(),
	# tf.keras.layers.MaxPool2D(pool_size=(2,2), padding="same"),

	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(512, activation='relu'),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Dense(2, activation='softmax')
])
