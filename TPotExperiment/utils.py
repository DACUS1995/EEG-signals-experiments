import numpy as np
import matplotlib.pyplot as plt
import librosa.display as display

def normalization(data):
    if type(data) != np.ndarray:
        raise Exception("Input data must be of type np.ndarray.")
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data - min_data) / (max_data - min_data  + 1e-6)
    return data


def plot_single_signal(signal):
	signal = np.copy(signal)
	if type(signal) != np.ndarray:
		raise Exception("Input data must be of type np.ndarray.")
	plt.figure()
	plt.plot(signal)
	plt.xticks(np.arange(signal.shape[0]))
	plt.show()


def plot_recording(data: np.ndarray):
	data = np.copy(data)
	if type(data) != np.ndarray:
		raise Exception("Input data must be of type np.ndarray.")
	
	for row_idx in range(data.shape[1]):
		data[:, row_idx] = data[:, row_idx] + row_idx

	plt.figure()
	for idx in range(data.shape[1]):
		plt.plot(data[:, idx])
	plt.xticks(np.arange(data.shape[0]))
	plt.show()

def plot_mfcc(mfccs):
	plt.figure(figsize=(10, 4))
	display.specshow(mfccs, x_axis='time')
	plt.colorbar()
	plt.title('MFCC')
	plt.tight_layout()
	plt.show()

def show_gabor(I, **kwargs):
	# utility function to show image
	plt.figure()
	plt.axis('off')
	plt.imshow(I, cmap=plt.gray(), **kwargs)
	plt.show()