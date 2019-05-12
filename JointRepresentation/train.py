import tensorflow as tf
import pathlib
from typing import List, Tuple, Dict 
import os
import pandas as pd
import numpy as np
import argparse
import datetime
import librosa

from config import Config
from models.model_1 import model as Model_1
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

def create_training_dataset(batch_size=5, shuffle=True, use_mfcc=True):
	recordings = []
	labels = []
	# dateset_file_paths = tf.data.Dataset.from_tensor_slices(get_recording_files_paths())

	for n, class_file_list in enumerate(get_recording_files_paths()):
		for m, file_path in enumerate(class_file_list):
			recordings.append(load_recording(file_path, use_mfcc))
			labels.append(n)

	dataset_recordings = tf.data.Dataset.from_tensor_slices(recordings)
	dataset_labels = tf.data.Dataset.from_tensor_slices(labels)
	dataset = tf.data.Dataset.zip((dataset_recordings, dataset_labels))

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
	output = tf.keras.layers.Flatten()(image_model_output)
	output = layers.Dense(embedding_size, activation='softmax')(output)

	model = tf.keras.Model(inputs=vgg.input, outputs=output)
	return


def train_step(img_tensor, eeg_signal, image_feature_extractor, eeg_feature_extractor):
	loss = 0

	with tf.GradientTape() as tape:
		features_image = image_feature_extractor(img_tensor)
		features_eeg = eeg_feature_extractor(eeg_signal)
		loss = tf.losses.cosine_distance(features_image, features_eeg)

	total_loss = loss
	trainable_variables = image_feature_extractor.trainable_variables
	gradients = tape.gradient(loss, trainable_variables)
	optimizer.apply_gradients(zip(gradients, trainable_variables))

	return loss, total_loss


def train(model, *, epochs=5, callbacks) -> None:
	dataset, length = create_training_dataset(batch_size=5)
	validation_dataset = dataset.take(int(Config.DATASET_TRAINING_VALIDATION_RATIO * length)) 
	train_dataset = dataset.skip(int(Config.DATASET_TRAINING_VALIDATION_RATIO * length))

	image_feature_extractor = create_image_encoder()
	eeg_feature_extractor = model

	# Checkpoint setup
	checkpoint_path = "./checkpoints/train"
	ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
	ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

	start_epoch = 0
	if ckpt_manager.latest_checkpoint:
		start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])

	optimizer = tf.keras.optimizers.Adam()
	loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

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
	if args.model == "model_1":
		model = Model_1
	if args.model == "model_mfcc":
		model = Model_mfcc

	checkpoint_prefix = os.path.join(Config.CHECKPOINTS_DIR, "ckpt_{epoch}")
	log_dir=Config.TENSORBOARD_LOGDIR + "\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


	# callbacks = [
	# 	# Interrupt training if `val_loss` stops improving for over 2 epochs
	# 	tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
	# 	tf.keras.callbacks.ModelCheckpoint(checkpoint_prefix, save_best_only=True, period=2),
	# 	# Write TensorBoard logs to `./logs` directory
	# 	tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
	# ]

	trained_model = train(model=model, epochs=args.epochs, callbacks=callbacks)
	if args.save_model == True:
		trained_model.save(f'{args.model}.h5')
		# new_model = keras.models.load_model('my_model.h5')

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs for training.")
	parser.add_argument("-m", "--model", type=str, default="model_1", help="What model to use.")
	args = parser.parse_args()
	main(args)
