import numpy as np


def normalization(data):
    if type(data) != np.ndarray:
        raise Exception("Input data must be of type np.ndarray.")
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data - min_data) / (max_data - min_data  + 1e-6)
    return data