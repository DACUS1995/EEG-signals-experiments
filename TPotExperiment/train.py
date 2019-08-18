import librosa
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict 
import os
import pandas as pd
import numpy as np
import argparse
import pathlib

from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import GridSearchCV
from scipy.signal import convolve
from tpot import TPOTClassifier

from config import Config
import utils

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


def compute_mfcc(recording):
	transformed_recording = None

	for i in range(recording.shape[1]):
		current_channel = librosa.feature.mfcc(recording[:, i], sr=160, n_mfcc=10, hop_length=10, n_fft=40)
		if transformed_recording is None:
			transformed_recording = current_channel
		else:
			transformed_recording = np.append(transformed_recording, current_channel, axis=0)

	assert transformed_recording is not None
	# print(transformed_recording.shape)
	return transformed_recording


def load_recording(path, use_mfcc=False):
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

	recording = np.transpose(recording)
	recording = recording.flatten()
	return recording
	# print(splitedFileContents)
	# return preprocess_file(splitedFileContents)


def create_training_dataset(shuffle=True, use_mfcc=False, use_gabor=False):
	recordings = []
	labels = []
	# dateset_file_paths = tf.data.Dataset.from_tensor_slices(get_recording_files_paths())

	gabor_filters = [genGabor((40, 1), omega=i) for i in np.arange(0.1, 1, 0.3)]
	print(f"Number of gabor filters used: [{len(gabor_filters)}]")

	for n, class_file_list in enumerate(get_recording_files_paths()):
		for m, file_path in enumerate(class_file_list):
			recording = load_recording(file_path, use_mfcc)

			if use_gabor == True:
				recording = np.empty((0), dtype=recording.dtype)
				for gabor in gabor_filters:
					recording = np.append(recording, convolve(recording, gabor, mode="valid"))
			
			recordings.append(recording)
			labels.append(n)

	return (recordings, labels)


def create_testing_dataset(use_mfcc=False, use_gabor=False):
	recordings = []
	labels = []

	gabor_filters = [genGabor((40, 1), omega=i) for i in np.arange(0.1, 1, 0.2)]

	for n, class_file_list in enumerate(get_recording_files_paths(mode="testing")):
		for m, file_path in enumerate(class_file_list):
			recording = load_recording(file_path, use_mfcc)

			if use_gabor == True:
				recording = np.empty((0), dtype=recording.dtype)
				for gabor in gabor_filters:
					recording = np.append(recording, convolve(recording, gabor, mode="valid"))
			
			recordings.append(recording)
			labels.append(n)

	return (recordings, labels)


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


def train(use_mfcc = False, use_gabor = False):
	# gabor_filters = [utils.plot_single_signal(genGabor((40, 1), omega=i)) for i in np.arange(0.4, 2, 0.3)]
	# gabor = genGabor((50, 1))
	# utils.plot_single_signal(gabor)

	x_train, y_train = create_training_dataset(use_mfcc=use_mfcc, use_gabor=False)
	x_test, y_test = create_testing_dataset(use_mfcc=use_mfcc, use_gabor=False)

	x_train = np.asarray(x_train)
	y_train = np.asarray(y_train)
	x_test = np.asarray(x_test)
	y_test = np.asarray(y_test)
	
	tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
	tpot.fit(x_train, y_train)
	print(tpot.score(x_test, y_test))
	tpot.export('tpot_mnist_pipeline.py')

def main(args):
	use_mfcc = False
	use_gabor = False
	train(use_mfcc=use_mfcc, use_gabor=use_gabor)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# parser.add_argument("-s", "--save_model", type=bool, default=True, help="Save model.")
	args = parser.parse_args()
	main(args)
