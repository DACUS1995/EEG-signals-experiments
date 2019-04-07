import tensorflow as tf
import pathlib
from typing import List, Tuple, Dict 
import os
import pandas as pd
import numpy as np

from config import Config

AUTOTUNE = tf.data.experimental.AUTOTUNE

try:
	tf.enable_eager_execution()
except:
	pass

print("Using eager execution: " + str(tf.executing_eagerly())) 
print("Using tensorflow version: " + str(tf.__version__) + "\n")


def get_recording_files_paths() -> List[str]:
	# data_root = os.path.join(Config.RECORDING_PATH_ROOT, "\Park\Surdoiu_Tudor\Day_1")
	data_root = Config.RECORDING_PATH_ROOT + "\Park\Surdoiu_Tudor\Day_1"
	data_root = pathlib.Path(data_root)
	all_recordings_path = list(data_root.glob('*'))

	print(f"Scanned [{len(all_recordings_path)}] images")

	all_recordings_path = [str(path) for path in all_recordings_path]
	return all_recordings_path

def preprocess_file(file):
	csv_file = tf.io.decode_csv(file, ["a"])
	return csv_file

def load_recording(path):
	# fileContents = tf.io.read_file(path)
	# splitedFileContents = tf.string_split([fileContents], os.linesep)
	df = pd.read_csv(path, skiprows=[0], header=None, names=["COUNTER", "INTERPOLATED", "F3", "FC5", "AF3", "F7", "T7", "P7", "O1", "O2", "P8", "T8", "F8", "AF4", "FC6", "F4", "RAW_CQ", "GYROX"]) # "GYROY", "MARKER", "MARKER_HARDWARE", "SYNC", "TIME_STAMP_s", "TIME_STAMP_ms", "CQ_AF3", "CQ_F7", "CQ_F3", "CQ_FC5", "CQ_T7", "CQ_P7", "CQ_O1", "CQ_O2", "CQ_P8", "CQ_T8", "CQ_FC6", "CQ_F4", "CQ_F8", "CQ_AF4", "CQ_CMS", "CQ_DRL"])

	df = df[:Config.RECORDING_NUM_SAMPLES]
	recording = df.values
	recording.dtype = np.float64

	if recording.shape[0] < Config.RECORDING_NUM_SAMPLES:
		recording = np.pad(recording, ((0, Config.RECORDING_NUM_SAMPLES - recording.shape[0]), (0, 0)), mode="edge")
	
	if recording.shape[0] != Config.RECORDING_NUM_SAMPLES:
		raise Exception(f"Session number of samples is super not OK: [{recording.shape[0]}]")

	return recording
	# print(splitedFileContents)
	# return preprocess_file(splitedFileContents)

recordings = []
dateset_file_paths = tf.data.Dataset.from_tensor_slices(get_recording_files_paths())
# for n, file_path in enumerate(dateset_file_paths.take(4)):
for n, file_path in enumerate(get_recording_files_paths()):
	recordings.append(load_recording(file_path))

print(recordings[0])

dataset_recordings = tf.data.Dataset.from_tensor_slices(recordings)

for n, recording in enumerate(dataset_recordings.take(1)):
	print(recording[0])


# mnist = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

# model = tf.keras.models.Sequential([
# 	tf.keras.layers.Flatten(input_shape=(28, 28)),
# 	tf.keras.layers.Dense(128, activation='relu'),
# 	tf.keras.layers.Dropout(0.2),
# 	tf.keras.layers.Dense(10, activation='softmax')
# ])

# model.compile(
# 	optimizer='adam',
# 	loss='sparse_categorical_crossentropy',
# 	metrics=['accuracy']
# )

# model.fit(x_train, y_train, epochs=5)

# model.evaluate(x_test, y_test)