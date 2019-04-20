import numpy as np
import matplotlib.pyplot as plt


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