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

from config import Config
from models.model_mfcc import model as Model_mfcc
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


def load_img(path_to_img):
	max_dim = 512
	img = tf.io.read_file(path_to_img)
	img = tf.image.decode_image(img, channels=3)
	img = tf.image.convert_image_dtype(img, tf.float32)

	shape = tf.cast(tf.shape(img)[:-1], tf.float32)
	long_dim = max(shape)
	scale = max_dim / long_dim

	new_shape = tf.cast(shape * scale, tf.int32)

	img = tf.image.resize(img, new_shape)
	img = img[tf.newaxis, :]
	return img


def load_and_process_img(path_to_img):
	img = load_img(path_to_img)
	img = tf.keras.applications.vgg19.preprocess_input(img)
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
	dataset = tf.data.Dataset.zip((dataset_img, dataset_recordings))

	#  !! Remeber to shuffle berfore using batch !!
	if shuffle == True:
		dataset = dataset.shuffle(len(recordings))

	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(buffer_size=AUTOTUNE)
	return dataset, len(recordings)

def create_testing_dataset(use_mfcc=True):
	recordings = []
	labels = []

	for n, class_file_list in enumerate(get_recording_files_paths()):
		for m, file_path in enumerate(class_file_list):
			recordings.append(load_recording(file_path, use_mfcc))
			labels.append(n)

	dataset_recordings = tf.data.Dataset.from_tensor_slices(recordings)
	dataset_labels = tf.data.Dataset.from_tensor_slices(labels)
	dataset = tf.data.Dataset.zip((dataset_recordings, dataset_labels))
	dataset = dataset.batch(1)
	dataset = dataset.shuffle(len(recordings))
	return dataset


def create_image_encoder(output_layer_name="block5_conv4", embedding_size=512):
	image_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
	image_model.trainable = False # !Maybe fine tune this!
	
	image_model_output = image_model.layers[-1].output
	output = tf.squeeze(image_model_output, axis=[1, 2])
	output = tf.keras.layers.Flatten()(output)
	output = tf.keras.layers.Dense(embedding_size, activation='softmax')(output)

	model = tf.keras.Model(inputs=image_model.input, outputs=output)
	return model


def train_step(img_tensor, eeg_signal, image_feature_extractor, eeg_feature_extractor):
	loss = 0
	img_tensor = tf.squeeze(img_tensor)
	
	with tf.GradientTape() as tape:
		features_image = image_feature_extractor(img_tensor)
		features_eeg = eeg_feature_extractor(eeg_signal)
		loss = tf.losses.cosine_distance(features_image, features_eeg)

	total_loss = loss
	trainable_variables = image_feature_extractor.trainable_variables
	gradients = tape.gradient(loss, trainable_variables)
	optimizer.apply_gradients(zip(gradients, trainable_variables))

	return loss, total_loss


def train(model, *, epochs=5) -> None:
	dataset, length = create_training_dataset(batch_size=5)
	validation_dataset = dataset.take(int(Config.DATASET_TRAINING_VALIDATION_RATIO * length)) 
	train_dataset = dataset.skip(int(Config.DATASET_TRAINING_VALIDATION_RATIO * length))

	image_feature_extractor = create_image_encoder()
	eeg_feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[10].output)
	optimizer = tf.keras.optimizers.Adam()

	# Checkpoint setup
	checkpoint_path = "./checkpoints/train"
	ckpt = tf.train.Checkpoint(encoder=image_feature_extractor, decoder=eeg_feature_extractor, optimizer=optimizer)
	ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

	start_epoch = 0
	if ckpt_manager.latest_checkpoint:
		start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])

	loss_plot = []

	for epoch in range(start_epoch, epochs):
		start = time.time()
		total_loss = 0

		for (batch, (img_tensor, record_sample)) in enumerate(dataset):
			batch_loss, t_loss = train_step(img_tensor, record_sample, image_feature_extractor, eeg_feature_extractor)
			total_loss += t_loss

			if batch % 100 == 0:
				print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))

		# storing the epoch end loss value to plot later
		loss_plot.append(total_loss / num_steps)

		if epoch % 5 == 0:
			ckpt_manager.save()

		print ('Epoch {} Loss {:.6f}'.format(epoch + 1, total_loss/num_steps))
		print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

	# model.fit(
	# 	train_dataset, 
	# 	epochs=epochs, 
	# 	validation_data=validation_dataset,
	# 	callbacks=callbacks
	# )

	dataset_test = create_testing_dataset()
	model.evaluate(dataset_test)
	return model
	# for n, (recording, label) in enumerate(dataset):
	# 	print(recording[0, 0])


def main(args):
	model = None
	if args.model == "model_mfcc":
		model = Model_mfcc

	model.compile(
		optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']
	)

	latest = tf.train.latest_checkpoint("./models/checkpoints_mfcc")
	model.load_weights(latest)
	model.trainable  = False

	checkpoint_prefix = os.path.join(Config.CHECKPOINTS_DIR, "ckpt_{epoch}")
	log_dir=Config.TENSORBOARD_LOGDIR + "\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


	# callbacks = [
	# 	# Interrupt training if `val_loss` stops improving for over 2 epochs
	# 	tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
	# 	tf.keras.callbacks.ModelCheckpoint(checkpoint_prefix, save_best_only=True, period=2),
	# 	# Write TensorBoard logs to `./logs` directory
	# 	tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
	# ]

	trained_model = train(model=model, epochs=args.epochs)
	if args.save_model == True:
		trained_model.save(f'{args.model}.h5')
		# new_model = keras.models.load_model('my_model.h5')

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs for training.")
	parser.add_argument("-m", "--model", type=str, default="model_mfcc", help="What model to use.")
	parser.add_argument("-s", "--save_model", type=bool, default=True, help="Save the model.")
	args = parser.parse_args()
	main(args)
