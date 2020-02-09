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
from models.dcgan import define_discriminator, define_generator, define_eeg_encoder
import utils
from frechet_inception_distance import calculate_fid

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

	new_shape = (112, 112)

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


def create_training_dataset(batch_size=5, shuffle=True, use_mfcc=True):
	recordings = []
	images = []
	labels = []
	# dateset_file_paths = tf.data.Dataset.from_tensor_slices(get_recording_files_paths())

	for n, class_file_list in enumerate(get_recording_files_paths()):
		for m, file_path in enumerate(class_file_list):
			recordings.append(load_recording(file_path, use_mfcc))
			path_components = file_path.split("\\")
			path_to_img = "D:\Storage\BrainImages\\" + path_components[5] + "\\" + path_components[6].replace("csv", "jpg")
			images.append(load_and_process_img(path_to_img))
			labels.append(one_hot_encode(n))


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

	#  !! Remeber to shuffle berfore using batch !!
	if shuffle == True:
		dataset = dataset.shuffle(len(recordings))

	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(buffer_size=AUTOTUNE)
	return dataset, len(recordings)


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


def discriminator_loss(real_output, fake_output):
	cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
	real_loss = cross_entropy(tf.zeros_like(real_output), real_output)
	fake_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
	total_loss = real_loss + fake_loss
	return total_loss

# We want the target to be ones(descriminator believe the gen samples to be true)
def generator_loss(fake_output):
	cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
	return cross_entropy(tf.zeros_like(fake_output), fake_output)


def plot_training_metrics(train_gen_loss_results, train_disc_loss_results):
	fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
	fig.suptitle('Training Metrics')

	axes[0].set_ylabel("Loss gen", fontsize=14)
	axes[0].set_xlabel("Epoch", fontsize=14)
	axes[0].plot(train_gen_loss_results)

	axes[1].set_ylabel("Loss disc", fontsize=14)
	axes[1].set_xlabel("Epoch", fontsize=14)
	axes[1].plot(train_disc_loss_results)
	plt.show()


def compute_accuracy(real_output, fake_output):
	real = tf.keras.metrics.BinaryAccuracy()
	fake = tf.keras.metrics.BinaryAccuracy()

	# print(real_output)
	# print(fake_output)

	real.update_state(tf.zeros_like(real_output), real_output)
	fake.update_state(tf.ones_like(fake_output), fake_output)

	return real.result(), fake.result()


# @tf.function
def train_step(
	images,
	record_sample,
	batch_size, 
	noise_dim, 
	generator_optimizer, 
	discriminator_optimizer,
	generator,
	discriminator
):
	noise = tf.random.normal([batch_size, noise_dim])

	images = tf.cast(images, tf.float32)
	record_sample = tf.cast(record_sample, tf.float32)

	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		generated_images = generator([noise, record_sample], training=True)

		real_output = discriminator([images, record_sample], training=True)
		fake_output = discriminator([generated_images, record_sample], training=True)

		gen_loss = generator_loss(fake_output)
		disc_loss = discriminator_loss(real_output, fake_output)

		real_acc, fake_acc = compute_accuracy(real_output, fake_output)

	gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
	gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

	generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
	discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


	return gen_loss, disc_loss, real_acc, fake_acc


def train(generator, discriminator, *, epochs=5, validation_dataset, train_dataset) -> None:
	generator_optimizer = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
	discriminator_optimizer = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)

	eeg_encoder = define_eeg_encoder()

	train_gen_loss_results, train_disc_loss_results = [], []

	log_dir = "logs/" + datetime.datetime.now().strftime("gan-%Y%m%d-%H%M%S")
	writer = tf.summary.create_file_writer(log_dir)
	
	with writer.as_default():
		with tf.summary.record_if(True):
			for epoch in range(epochs):
				epoch_loss_gen_avg = tf.keras.metrics.Mean()
				epoch_loss_disc_avg = tf.keras.metrics.Mean()

				epoch_acc_true_avg = tf.keras.metrics.Mean()
				epoch_acc_fake_avg = tf.keras.metrics.Mean()

				start = time.time()

				for (batch, (record_sample, img_tensor, label)) in enumerate(train_dataset):
					record_sample = eeg_encoder(tf.cast(record_sample, tf.float32))

					gen_loss, disc_loss, real_acc, fake_acc = train_step(
						img_tensor,
						record_sample,
						batch_size = record_sample.shape[0],
						noise_dim = 100,
						generator_optimizer = generator_optimizer,
						discriminator_optimizer = discriminator_optimizer,
						generator = generator,
						discriminator = discriminator
					)
					epoch_loss_gen_avg(gen_loss)
					epoch_loss_disc_avg(disc_loss)

					epoch_acc_true_avg(real_acc)
					epoch_acc_fake_avg(fake_acc)
				
				train_gen_loss_results.append(epoch_loss_gen_avg.result())
				train_disc_loss_results.append(epoch_loss_disc_avg.result())

				tf.summary.scalar('loss gen', epoch_loss_gen_avg.result(), step=epoch)
				tf.summary.scalar('loss disc', epoch_loss_disc_avg.result(), step=epoch)

				tf.summary.scalar('fake acc', epoch_acc_fake_avg.result(), step=epoch)
				tf.summary.scalar('real acc', epoch_acc_true_avg.result(), step=epoch)
				
				noise = tf.random.normal([record_sample.shape[0], 100])
				reconstructed = generator([noise, record_sample])
				original = tf.reshape(img_tensor, (-1, 112, 112, 3))

				tf.summary.image('original', original, max_outputs=100, step=epoch)
				tf.summary.image('reconstructed', reconstructed, max_outputs=100, step=epoch)


				# Save the model every 15 epochs
				if (epoch + 1) % 5 == 0:
					pass

				print("Epoch {:03d}: Loss generator: {:.3f}".format(epoch, epoch_loss_gen_avg.result()))
				print("Epoch {:03d}: Loss discriminator: {:.3f}".format(epoch, epoch_loss_disc_avg.result()))
				print ('Time for epoch {} is {} sec \n'.format(epoch + 1, time.time()-start))

	plot_training_metrics(train_gen_loss_results, train_disc_loss_results)

	return generator, discriminator

def main(args):
	dataset, length = create_training_dataset(batch_size=5)
	train_dataset = dataset

	testing_dataset, length = create_testing_dataset()
	validation_dataset = testing_dataset.take(5)

	generator = define_generator()
	discriminator = define_discriminator()

	generator, discriminator = train(
		generator=generator,
		discriminator=discriminator,
		epochs=args.epochs,
		validation_dataset=validation_dataset,
		train_dataset=train_dataset
	)

	noise = tf.random.normal([5, 100])
	plt.figure(figsize=(5, 4))
	for (batch, (record_sample, img_tensor, label)) in enumerate(validation_dataset):
		record_sample = tf.cast(record_sample, tf.float32)
		eeg_encoder = define_eeg_encoder()

		reconstructed = generator([noise, eeg_encoder(record_sample)])
		original = tf.reshape(img_tensor, (-1, 112, 112, 3))

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

		calculate_fid(original, reconstructed)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs for training.")
	parser.add_argument("-s", "--save_model", type=bool, default=True, help="Save the model.")
	args = parser.parse_args()
	main(args)
