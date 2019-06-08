import tensorflow as tf
import pathlib
from typing import List, Tuple, Dict 
import os
import pandas as pd
import numpy as np
import argparse
import librosa

from models.model_1 import model as Model_1
from models.model_mfcc import model as Model_mfcc
import utils
from config import Config

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


def load_recording(path, use_mfcc=False):
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

	return recording
	# print(splitedFileContents)
	# return preprocess_file(splitedFileContents)


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

def create_testing_dataset(use_mfcc=True):
	recordings = []
	labels = []

	for n, class_file_list in enumerate(get_recording_files_paths(mode="training")):
		for m, file_path in enumerate(class_file_list):
			recordings.append(load_recording(file_path, use_mfcc))
			labels.append(n)

	dataset_recordings = tf.data.Dataset.from_tensor_slices(recordings)
	dataset_labels = tf.data.Dataset.from_tensor_slices(labels)
	dataset = tf.data.Dataset.zip((dataset_recordings, dataset_labels))
	dataset = dataset.batch(1)
	dataset = dataset.shuffle(len(recordings))
	return dataset


def main(args):
	use_mfcc = False
	model = None
	
	if args.model == "model_1":
		model = Model_1
	if args.model == "model_mfcc":
		use_mfcc = True
		model = Model_mfcc

	model.compile(
		optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']
	)

	latest = tf.train.latest_checkpoint("./checkpoints")
	model.load_weights(latest)

	dataset_test = create_testing_dataset(use_mfcc)
	model.evaluate(dataset_test)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--model", type=str, default="model_mfcc", help="What model to use.")
	args = parser.parse_args()
	main(args)