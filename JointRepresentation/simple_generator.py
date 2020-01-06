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
from models.generator import Autoencoder
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
	img = tf.image.rgb_to_grayscale(img)

	new_shape = (56, 56)

	img = tf.image.resize(img, new_shape)
	# img = img[tf.newaxis, :]
	return img


def load_and_process_img(path_to_img):
	img = load_img(path_to_img)
	img = tf.reshape(img, (56 * 56,))
	# img = tf.keras.applications.vgg19.preprocess_input(img)
	return img


def create_training_dataset(batch_size=5, shuffle=True, use_mfcc=True):
	recordings = []
	images = []
	# dateset_file_paths = tf.data.Dataset.from_tensor_slices(get_recording_files_paths())

	for n, class_file_list in enumerate(get_recording_files_paths()):
		for m, file_path in enumerate(class_file_list):
			recordings.append(load_recording(file_path, use_mfcc))
			path_components = file_path.split("\\")
			path_to_img = "D:\Storage\BrainImages\\" + path_components[5] + "\\" + path_components[6].replace("csv", "jpg")
			images.append(load_and_process_img(path_to_img))

	dataset_img = tf.data.Dataset.from_tensor_slices(images)
	dataset_recordings = tf.data.Dataset.from_tensor_slices(recordings)
	dataset = tf.data.Dataset.zip((dataset_recordings, dataset_img))

	#  !! Remeber to shuffle berfore using batch !!
	if shuffle == True:
		dataset = dataset.shuffle(len(recordings))

	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(buffer_size=AUTOTUNE)
	return dataset, len(recordings)

def create_testing_dataset(batch_size=5, use_mfcc=True):
	recordings = []
	images = []

	for n, class_file_list in enumerate(get_recording_files_paths(mode="testing")):
		for m, file_path in enumerate(class_file_list):
			recordings.append(load_recording(file_path, use_mfcc))
			path_components = file_path.split("\\")
			path_to_img = "D:\Storage\BrainImages\\" + path_components[5] + "\\" + path_components[6].replace("csv", "jpg")
			images.append(load_and_process_img(path_to_img))

	dataset_img = tf.data.Dataset.from_tensor_slices(images)
	dataset_recordings = tf.data.Dataset.from_tensor_slices(recordings)
	dataset = tf.data.Dataset.zip((dataset_recordings, dataset_img))

	dataset = dataset.shuffle(len(recordings))

	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(buffer_size=AUTOTUNE)
	return dataset, len(recordings)


def plot_training_metrics(train_loss_results):
	fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
	fig.suptitle('Training Metrics')

	axes[0].set_ylabel("Loss", fontsize=14)
	axes[0].set_xlabel("Epoch", fontsize=14)
	axes[0].plot(train_loss_results)
	plt.show()


def create_model(dataset):
	final_model = Autoencoder(
		intermediate_dim=512, 
		original_dim=3136,
		dataset=dataset
	)

	return final_model


def grad(model, record_sample, img_tensor):
	with tf.GradientTape() as tape:
		loss_value = loss(model, record_sample, img_tensor)

	return loss_value, tape.gradient(loss_value, model.trainable_variables)

def loss(model, record_sample, img_tensor):
	# print(record_sample.dtype)
	# print(record_sample.shape)
	# print(img_tensor.dtype)
	# print(img_tensor.shape)

	img_tensor = tf.cast(img_tensor, tf.float32)
	record_sample = tf.cast(record_sample, tf.float32)

	reconstructed = tf.reshape(model(record_sample), (-1, 56, 56, 1))
	original = tf.reshape(img_tensor, (-1, 56, 56, 1))

	reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(reconstructed, original)))
	return reconstruction_error

def train(model, *, epochs=5, validation_dataset, train_dataset) -> None:
	optimizer = tf.keras.optimizers.Adam()

	start_epoch = 0
	train_loss_results = []

	log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	writer = tf.summary.create_file_writer(log_dir)
	
	with writer.as_default():
		with tf.summary.record_if(True):
			for epoch in range(start_epoch, epochs):
				epoch_loss_avg = tf.keras.metrics.Mean()

				for (batch, (record_sample, img_tensor)) in enumerate(train_dataset):
					# Optimize the model
					loss_value, grads = grad(model, record_sample, img_tensor)
					optimizer.apply_gradients(zip(grads, model.trainable_variables))

					# Track progress
					epoch_loss_avg(loss_value)  # Add current batch loss

				# End epoch
				train_loss_results.append(epoch_loss_avg.result())

				if epoch % 1 == 0:
					print("Epoch {:03d}: Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))

				tf.summary.scalar('loss', loss_value, step=epoch)

				if epoch % 5 == 0:
					record_sample = tf.cast(record_sample, tf.float32)
					reconstructed = tf.reshape(model(record_sample), (-1, 56, 56, 1))
					original = tf.reshape(img_tensor, (-1, 56, 56, 1))

					tf.summary.image('original', original, max_outputs=100, step=epoch)
					tf.summary.image('reconstructed', reconstructed, max_outputs=100, step=epoch)
			
		
	plot_training_metrics(train_loss_results)

	return model


def main(args):
	dataset, length = create_training_dataset(batch_size=5)
	validation_dataset = dataset.take(int(Config.DATASET_TRAINING_VALIDATION_RATIO * length)) 
	train_dataset = dataset.skip(int(Config.DATASET_TRAINING_VALIDATION_RATIO * length))

	model = create_model(dataset)

	trained_model = train(
		model=model,
		epochs=args.epochs,
		validation_dataset=validation_dataset,
		train_dataset=train_dataset
	)

	dataset_test, length = create_testing_dataset()

	plt.figure(figsize=(5, 4))
	for (batch, (record_sample, img_tensor)) in enumerate(dataset_test.take(5)):
		record_sample = tf.cast(record_sample, tf.float32)

		reconstructed = tf.reshape(trained_model(record_sample), (-1, 56, 56))
		original = tf.reshape(img_tensor, (-1, 56, 56))

		reconstructed = reconstructed.numpy()
		original = original.numpy()


		for index in range(5):
			# display original
			ax = plt.subplot(2, 5, index + 1)
			test_image = original[index]
			plt.imshow(test_image)
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
		
			# display reconstruction
			ax = plt.subplot(2, 5, index + 1 + 5)
			created_image = reconstructed[index]
			plt.imshow(created_image)
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
		
		plt.show()


		# plt.imshow(np.squeeze(reconstructed[0]), cmap='gray')
		# plt.show()
		# plt.imshow(np.squeeze(original[0]), cmap='gray')
		# plt.show()


	if args.save_model == True:
		trained_model.save_weights('./generator.h5')
		# new_model = keras.models.load_model('my_model.h5')

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs for training.")
	parser.add_argument("-s", "--save_model", type=bool, default=True, help="Save the model.")
	args = parser.parse_args()
	main(args)
