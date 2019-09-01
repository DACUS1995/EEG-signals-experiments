import tensorflow as tf
import pathlib
from typing import List, Tuple, Dict 
import os
import pandas as pd
import numpy as np
import argparse
import datetime
import librosa
import matplotlib.pyplot as plt
from scipy.signal import convolve

from config import Config
from models.model_1 import model as Model_1
from models.model_mfcc import model as Model_mfcc
from models.model_RNN import model as Model_lstm
import utils

AUTOTUNE = tf.data.experimental.AUTOTUNE

try:
	tf.enable_eager_execution()
except:
	pass

print("Using eager execution: " + str(tf.executing_eagerly())) 
print("Using tensorflow version: " + str(tf.__version__) + "\n")


def get_recording_files_paths(mode="training") -> List[List]:
	# data_root = os.path.join(Config.RECORDING_PATH_ROOT, "\Park\Surdoiu_Tudor\Day_1")
	if mode == "training":
		data_root = Config.RECORDING_PATH_ROOT + "\\train"
	if mode == "testing":
		data_root = Config.RECORDING_PATH_ROOT + "\\test"

	
		
	data_root = pathlib.Path(data_root)
	classes_dir = list(data_root.glob('*'))
	print(f"Number of classes directories: [{len(classes_dir)}]")

	all_file_paths = []
	number_of_samples = 0

	for class_dir in classes_dir:
		recording_files = list(pathlib.Path(class_dir).glob('*'))
		all_file_paths.append([str(path) for path in recording_files])
		number_of_samples += len(recording_files)

	print(f"Scanned [{number_of_samples}] images")
	return all_file_paths

def preprocess_file(file):
	csv_file = tf.io.decode_csv(file, ["a"])
	return csv_file

def compute_mfcc(recording):
	transformed_recording = None

	for i in range(recording.shape[1]):
		current_channel = librosa.feature.mfcc(recording[:, i], sr=40, n_mfcc=10, hop_length=10, n_fft=40)
		if transformed_recording is None:
			transformed_recording = current_channel
		else:
			transformed_recording = np.append(transformed_recording, current_channel, axis=0)

	assert transformed_recording is not None
	# print(transformed_recording.shape)
	return transformed_recording

def load_recording(path, use_mfcc=False, use_gabor=False):
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

	if use_mfcc:
		recording = compute_mfcc(recording)

	res = recording

	if(use_gabor):
		recording = np.transpose(recording)
		gabor_filters = [genGabor((40, 1), omega=i) for i in np.arange(0.1, 1, 0.2)]
		# res = np.empty((0), dtype=recording.dtype)
		res = []

		for gabor in gabor_filters:
			for line in range(recording.shape[0]):
				res.append(convolve(recording[line], gabor, mode="same"))

		res = np.transpose(np.array(res))

	return res


def create_training_dataset(batch_size=5, shuffle=True, use_mfcc=True, use_gabor=False):
	recordings = []
	labels = []
	# dateset_file_paths = tf.data.Dataset.from_tensor_slices(get_recording_files_paths())

	for n, class_file_list in enumerate(get_recording_files_paths()):
		for m, file_path in enumerate(class_file_list):
			recordings.append(load_recording(file_path, use_mfcc, use_gabor))
			# print(load_recording(file_path, use_mfcc).shape)
			labels.append(n)

	dataset_recordings = tf.data.Dataset.from_tensor_slices(recordings)
	dataset_labels = tf.data.Dataset.from_tensor_slices(labels)
	dataset = tf.data.Dataset.zip((dataset_recordings, dataset_labels))

	# for n, (recording, label) in enumerate(dataset.take(1)):
	# 	print(tf.math.reduce_max(recording, axis=0))
	# 	print(tf.math.reduce_min(recording, axis=0))
	# 	print(recording)

	#  !! Remember to shuffle berfore using batch !!
	if shuffle == True:
		dataset = dataset.shuffle(len(recordings))

	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(buffer_size=AUTOTUNE)
	return dataset, len(recordings)

def create_testing_dataset(use_mfcc=True, use_gabor=False):
	recordings = []
	labels = []

	for n, class_file_list in enumerate(get_recording_files_paths(mode="testing")):
		for m, file_path in enumerate(class_file_list):
			recordings.append(load_recording(file_path, use_mfcc, use_gabor))
			labels.append(n)

	dataset_recordings = tf.data.Dataset.from_tensor_slices(recordings)
	dataset_labels = tf.data.Dataset.from_tensor_slices(labels)
	dataset = tf.data.Dataset.zip((dataset_recordings, dataset_labels))
	dataset = dataset.batch(1)
	dataset = dataset.shuffle(len(recordings))
	return dataset, len(recordings)


def genGabor(sz, omega=0.5, theta=0, func=np.cos, K=np.pi):
	radius = (int(sz[0]/2.0), int(sz[1]/2.0))
	[x, y] = np.meshgrid(range(-radius[0], radius[0]+1), range(-radius[1], radius[1]+1))

	x1 = x * np.cos(theta) + y * np.sin(theta)
	y1 = -x * np.sin(theta) + y * np.cos(theta)
    
	gauss = omega**2 / (4*np.pi * K**2) * np.exp(- omega**2 / (8*K**2) * ( 4 * x1**2 + y1**2))
	sinusoid = func(omega * x1) * np.exp(K**2 / 2)
	gabor = gauss * sinusoid
	gabor = np.array(gabor)
	return gabor.flatten()


def train(model, *, epochs=5, callbacks, use_mfcc, use_gabor) -> None:
	dataset, length = create_training_dataset(batch_size=5, use_mfcc=use_mfcc, use_gabor=use_gabor)
	dataset_test, length_test = create_testing_dataset(use_mfcc=use_mfcc, use_gabor=use_gabor)

	# Use samples from the same session for validation
	# validation_dataset = dataset.take(int(Config.DATASET_TRAINING_VALIDATION_RATIO * length)) 
	# train_dataset = dataset.skip(int(Config.DATASET_TRAINING_VALIDATION_RATIO * length))

	# Use samples from different session for validation
	validation_dataset = dataset_test.take(int(Config.DATASET_TRAINING_VALIDATION_RATIO * length_test)) 
	dataset_test = dataset_test.skip(int(Config.DATASET_TRAINING_VALIDATION_RATIO * length_test))
	train_dataset = dataset

	model.fit(
		train_dataset, 
		epochs=epochs, 
		validation_data=validation_dataset,
		callbacks=callbacks
	)

	model.evaluate(dataset_test)
	return model
	# for n, (recording, label) in enumerate(dataset):
	# 	print(recording[0, 0])


def main(args):
	model = None
	use_mfcc = False
	use_gabor = False

	if args.model == "model_1":
		model = Model_1
	if args.model == "model_mfcc":
		use_mfcc = True
		model = Model_mfcc
	if args.model == "model_lstm":
		model = Model_lstm

	checkpoint_prefix = os.path.join(Config.CHECKPOINTS_DIR, "ckpt_{epoch}")
	log_dir=Config.TENSORBOARD_LOGDIR + "\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


	callbacks = [
		# Interrupt training if `val_loss` stops improving for over 2 epochs
		# tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss'),
		tf.keras.callbacks.ModelCheckpoint(checkpoint_prefix, save_best_only=True, period=2),
		# Write TensorBoard logs to `./logs` directory
		tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
	]

	model.compile(
		optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']
	)

	trained_model = train(
		model=model, 
		epochs=args.epochs, 
		callbacks=callbacks,
		use_mfcc=use_mfcc,
		use_gabor=use_gabor
	)
	# if args.save_model == True:
	# 	trained_model.save_weights(f'{args.model}.h5')
		# new_model = keras.models.load_model('my_model.h5')

	if args.print_summary == True:
		model.summary()
		utils.plot_model(model)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs for training.")
	parser.add_argument("-m", "--model", type=str, default="model_1", help="What model to use.")
	parser.add_argument("-p", "--print_summary", type=bool, default=False, help="Print summary and plot model")
	# parser.add_argument("-s", "--save_model", type=bool, default=True, help="Save model.")
	args = parser.parse_args()
	main(args)
