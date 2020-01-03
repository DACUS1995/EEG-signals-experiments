class Config:
	RECORDING_PATH_ROOT = "D:\Storage\EEGRecordings\Part_two\\"
	RECORDING_NUM_SAMPLES = 190
	SENSORS_LABELS = ["F3", "FC5", "AF3", "F7", "T7", "P7", "O1", "O2", "P8", "T8", "F8", "AF4", "FC6", "F4"]
	DATASET_TRAINING_VALIDATION_RATIO = 0.1
	CHECKPOINTS_DIR = './checkpoints'
	TENSORBOARD_LOGDIR = "logs"