import tensorflow as tf
import pathlib
from typing import List, Tuple, Dict 
import os
import pandas as pd
import numpy as np
import argparse

from config import Config
from models.model_1 import model as Model_1
import utils

AUTOTUNE = tf.data.experimental.AUTOTUNE

try:
	tf.enable_eager_execution()
except:
	pass

print("Using eager execution: " + str(tf.executing_eagerly())) 
print("Using tensorflow version: " + str(tf.__version__) + "\n")


def get_recording_files_paths(mode="training") -> List[str]:
	# data_root = os.path.join(Config.RECORDING_PATH_ROOT, "\Park\Surdoiu_Tudor\Day_1")
	if mode == "training":
		data_root = Config.RECORDING_PATH_ROOT + "\Park\Surdoiu_Tudor\Day_1"
	if mode == "testing":
		data_root = Config.RECORDING_PATH_ROOT + "\Park\Surdoiu_Tudor\Day_2"
		
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
	df = df[Config.SENSORS_LABELS]
	recording = df.values
	recording.dtype = np.float64
	recording = utils.normalization(recording)

	if recording.shape[0] < Config.RECORDING_NUM_SAMPLES:
		recording = np.pad(recording, ((0, Config.RECORDING_NUM_SAMPLES - recording.shape[0]), (0, 0)), mode="edge")
	
	if recording.shape[0] != Config.RECORDING_NUM_SAMPLES:
		raise Exception(f"Session number of samples is super not OK: [{recording.shape[0]}]")

	return recording
	# print(splitedFileContents)
	# return preprocess_file(splitedFileContents)

def create_training_dataset(batch_size=5, shuffle=True):
	recordings = []
	labels = []
	# dateset_file_paths = tf.data.Dataset.from_tensor_slices(get_recording_files_paths())

	for n, file_path in enumerate(get_recording_files_paths()):
		recordings.append(load_recording(file_path))
		labels.append(0)
		recordings.append(np.random.rand(Config.RECORDING_NUM_SAMPLES, len(Config.SENSORS_LABELS)))
		labels.append(1)

	dataset_recordings = tf.data.Dataset.from_tensor_slices(recordings)
	dataset_labels = tf.data.Dataset.from_tensor_slices(labels)
	dataset = tf.data.Dataset.zip((dataset_recordings, dataset_labels))
	# for n, (recording, label) in enumerate(dataset.take(1)):
	# 	print(label)

	#  !! Remeber to shuffle berfore using batch !!
	if shuffle == True:
		dataset = dataset.shuffle(2 * len(recordings))

	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(buffer_size=AUTOTUNE)
	return dataset, len(recordings)

def create_testing_dataset():
	recordings = []
	labels = []

	for n, file_path in enumerate(get_recording_files_paths(mode="testing")):
		recordings.append(load_recording(file_path))
		labels.append(0)
		recordings.append(np.random.rand(Config.RECORDING_NUM_SAMPLES, len(Config.SENSORS_LABELS)))
		labels.append(1)

	dataset_recordings = tf.data.Dataset.from_tensor_slices(recordings)
	dataset_labels = tf.data.Dataset.from_tensor_slices(labels)
	dataset = tf.data.Dataset.zip((dataset_recordings, dataset_labels))
	return dataset


def train(model, epochs=5):
	dataset, length = create_training_dataset(batch_size=10)
	validation_dataset = dataset.take(int(Config.DATASET_TRAINING_VALIDATION_RATIO * length)) 
	train_dataset = dataset.skip(int(Config.DATASET_TRAINING_VALIDATION_RATIO * length))
	model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset)

	dataset_test = create_testing_dataset()
	model.evaluate(dataset_test)
	# for n, (recording, label) in enumerate(dataset):
	# 	print(recording[0, 0])


def main(args):
	model = None
	if args.model == "model_1":
		model = Model_1

	callbacks = [
		# Interrupt training if `val_loss` stops improving for over 2 epochs
		tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
		tf.keras.callbacks.ModelCheckpoint("./", save_best_only=True, period=2)
		# Write TensorBoard logs to `./logs` directory
		# tf.keras.callbacks.TensorBoard(log_dir='./logs')
	]

	model.compile(
		optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']
	)
	train(model, args.epochs)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs for training.")
	parser.add_argument("-m", "--model", type=str, default="model_1", help="What model to use.")
	args = parser.parse_args()
	main(args)
