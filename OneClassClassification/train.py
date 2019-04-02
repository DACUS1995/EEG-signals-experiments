import tensorflow as tf
import pathlib
from typing import List, Tuple, Dict 

AUTOTUNE = tf.data.experimental.AUTOTUNE

try:
	tf.enable_eager_execution()
except:
	pass

print("Using eager execution: " + str(tf.executing_eagerly())) 
print("Using tensorflow version: " + str(tf.__version__) + "\n")

def get_recording_files_names() -> List[str]:
	data_root = "D:\Storage\EEGRecordings\Park\Surdoiu_Tudor\Day_1" #TODO put this in a config file
	data_root = pathlib.Path(data_root)
	all_recordings_path = list(data_root.glob('*'))

	print(f"Scanned [{len(all_recordings_path)}] images")

	all_recordings_path = [str(path) for path in all_recordings_path]
	return all_recordings_path

def preprocess_file(file):
	csv_file = tf.io.decode_csv(file, ["a"])
	return csv_file

def load_recording(path):
	file = tf.io.read_file(path)
	return preprocess_file(file)

dateset_file_paths = tf.data.Dataset.from_tensor_slices(get_recording_files_names())
dataset_recordings = dateset_file_paths.map(load_recording, num_parallel_calls=AUTOTUNE)

for n, recoding in enumerate(dataset_recordings.take(4)):
	print(recording.shape)
# mnist = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

# model = tf.keras.models.Sequential([
# 	tf.keras.layers.Flatten(input_shape=(28, 28)),
# 	tf.keras.layers.Dense(128, activation='relu'),
# 	tf.keras.layers.Dropout(0.2),
# 	tf.keras.layers.Dense(10, activation='softmax')
# ])

# model.compile(
# 	optimizer='adam',
# 	loss='sparse_categorical_crossentropy',
# 	metrics=['accuracy']
# )

# model.fit(x_train, y_train, epochs=5)

# model.evaluate(x_test, y_test)