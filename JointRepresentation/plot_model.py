import tensorflow as tf
from models.model_lstm import Model as Model_lstm
import utils

import argparse


def main(args):
	if args.model is not None:
		model_class = Model_lstm
		model = model_class()
	elif args.file is not None:
		model = tf.keras.models.load_model(f"models/{args.file}.h5")
	else:
		raise Exception("No valid model")

	model.summary()
	utils.plot_model(model, args.file)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--model", type=str, default=None, help="Name of the model to draw.")
	parser.add_argument("-f", "--file", type=str, default=None, help="Name of saved model.")
	args = parser.parse_args()
	main(args)


