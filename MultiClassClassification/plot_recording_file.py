import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd

import utils
from config import Config

def main(args):
	if args.file == None or args.file.split(".")[-1] != "csv":
		raise Exception("Invalid input file.")

	file_name = args.file
	file_path = Config.RECORDING_PATH_ROOT + "train\herastrau_train\\" + file_name

	df = pd.read_csv(file_path, skiprows=[0], header=None, names=["COUNTER", "INTERPOLATED", "F3", "FC5", "AF3", "F7", "T7", "P7", "O1", "O2", "P8", "T8", "F8", "AF4", "FC6", "F4", "RAW_CQ", "GYROX"]) # "GYROY", "MARKER", "MARKER_HARDWARE", "SYNC", "TIME_STAMP_s", "TIME_STAMP_ms", "CQ_AF3", "CQ_F7", "CQ_F3", "CQ_FC5", "CQ_T7", "CQ_P7", "CQ_O1", "CQ_O2", "CQ_P8", "CQ_T8", "CQ_FC6", "CQ_F4", "CQ_F8", "CQ_AF4", "CQ_CMS", "CQ_DRL"])
	df = df[Config.SENSORS_LABELS]
	recording = df.values
	recording.dtype = np.float64
	recording = utils.normalization(recording)
	utils.plot_recording(recording)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-f", "--file", type=str, default=None, help="Name of the EEG file recording.")
	args = parser.parse_args()
	main(args)
