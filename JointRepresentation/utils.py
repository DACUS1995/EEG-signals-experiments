import numpy as np
import matplotlib.pyplot as plt
import librosa.display as display
import tensorflow as tf

def normalization(data):
    if type(data) != np.ndarray:
        raise Exception("Input data must be of type np.ndarray.")
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data - min_data) / (max_data - min_data  + 1e-6)
    return data

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

def plot_model(model: tf.keras.Model, model_name = "model"):
	if not(isinstance(model, tf.keras.Model)):
		raise Exception("Model must be instance of tf.keras.Model!")

	tf.keras.utils.plot_model(
		model,
		to_file=f"{model_name}.png",
		show_shapes=True,
		show_layer_names=True,
		rankdir='TD'
	)