import tensorflow as tf
import pathlib
from typing import List, Tuple, Dict 
import os
import pandas as pd
import numpy as np
import argparse
import datetime
import librosa
import time
import matplotlib.pyplot as plt

from config import Config
from models.model_mfcc import Model as Model_mfcc
from models.model_lstm import Model as Model_lstm
from models.variational_autoencoder import Autoencoder
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
		data_root = Config.RECORDING_PATH_ROOT + "\\Session_1"
	if mode == "testing":
		data_root = Config.RECORDING_PATH_ROOT + "\\Session_2"

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

	return recording
	# print(splitedFileContents)
	# return preprocess_file(splitedFileContents)


def load_img(path_to_img):
	max_dim = 512
	img = tf.io.read_file(path_to_img)
	img = tf.image.decode_image(img, channels=3)
	img = tf.image.convert_image_dtype(img, tf.float32)
	# img = tf.image.rgb_to_grayscale(img)

	new_shape = (56, 56)

	img = tf.image.resize(img, new_shape)
	# img = img[tf.newaxis, :]
	return img


def load_and_process_img(path_to_img):
	img = load_img(path_to_img)
	# img = tf.reshape(img, (56 * 56,))
	# img = tf.keras.applications.vgg19.preprocess_input(img)
	return img

def one_hot_encode(class_num):
	if class_num == 0:
		return [1, 0]
	else:
		return [0, 1]


def create_testing_dataset(batch_size=5, use_mfcc=True):
	recordings = []
	images = []
	labels = []

	for n, class_file_list in enumerate(get_recording_files_paths(mode="testing")):
		for m, file_path in enumerate(class_file_list):
			recordings.append(load_recording(file_path, use_mfcc))
			path_components = file_path.split("\\")
			path_to_img = "D:\Storage\BrainImages\\" + path_components[5] + "\\" + path_components[6].replace("csv", "jpg")
			images.append(load_and_process_img(path_to_img))
			labels.append(one_hot_encode(n))

	dataset_img = tf.data.Dataset.from_tensor_slices(images)
	dataset_recordings = tf.data.Dataset.from_tensor_slices(recordings)
	dataset_labels = tf.data.Dataset.from_tensor_slices(labels)
	dataset = tf.data.Dataset.zip((dataset_recordings, dataset_img, dataset_labels))

	dataset = dataset.shuffle(len(recordings))

	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(buffer_size=AUTOTUNE)
	return dataset, len(recordings)



def create_model(dataset):
	final_model = Autoencoder(
		intermediate_dim=512, 
		original_dim=9408,
		dataset=dataset
	)

	return final_model

def main(args):
	testing_dataset, length = create_testing_dataset()
	validation_dataset = testing_dataset.take(5)

	# validation_dataset = dataset.take(int(Config.DATASET_TRAINING_VALIDATION_RATIO * length)) 
	# train_dataset = dataset.skip(int(Config.DATASET_TRAINING_VALIDATION_RATIO * length))

	model = create_model(validation_dataset)
	model.save_weights('./vae_generator.h5')

	random_vector_for_generation = tf.random.normal(
		shape=[1, 512]
	)

	vector_for_generation = tf.concat(
		[random_vector_for_generation, tf.convert_to_tensor([[1,0]], dtype=tf.float32)], 
		axis=1
	)

	reconstructed = tf.reshape(model.sample(vector_for_generation), (56, 56, 3)).numpy()
	plt.imshow(reconstructed)
	plt.show()


	plt.figure(figsize=(5, 4))
	for (batch, (record_sample, img_tensor, label)) in enumerate(validation_dataset.take(5)):
		record_sample = tf.cast(record_sample, tf.float32)
		label = tf.cast(label, tf.float32)

		reconstructed = tf.reshape(model(record_sample, label), (-1, 56, 56, 3))
		original = tf.reshape(img_tensor, (-1, 56, 56, 3))

		reconstructed = reconstructed.numpy()
		original = original.numpy()

		for index in range(5):
			# display original
			ax = plt.subplot(2, 5, index + 1)
			test_image = original[index]
			plt.imshow(test_image)
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
		
			# display reconstruction
			ax = plt.subplot(2, 5, index + 1 + 5)
			created_image = reconstructed[index]
			plt.imshow(created_image)
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
		
		plt.show()


		# plt.imshow(np.squeeze(reconstructed[0]), cmap='gray')
		# plt.show()
		# plt.imshow(np.squeeze(original[0]), cmap='gray')
		# plt.show()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	args = parser.parse_args()
	main(args)
