###################################################
# USEFUL IMPORTS:
# import datetime
# import numpy as np
# import pandas as pd

# import scipy as sp
# import matplotlib.pyplot as plt##
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import classification_report         
# from tensorflow.examples.tutorials.mnist import input_data
# from gensim import corpora, models, similarities
#################################################

import datetime
import time

START_TIME = time.time()

#################################################
# DIRECTORIES
CIFAR_DIR = '../../datasets/cifar-10-batches-py/'
SRC_DIR = "../src"

START_TIME_F = "{}".format(datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S"))
TIMESTEMP = "{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

TEMP_DIR = "../temp/".format(TIMESTEMP)
TEMP_DIR_SRC = TEMP_DIR + "src_{}/".format(TIMESTEMP)
LOGS_DIR = "../logs/"
TENSORBOARD_PATH = LOGS_DIR + "TB"
SAVED_MODELS_DIR = "../saved_models/"
MODEL_DIR = SAVED_MODELS_DIR + "Model_{}_".format(TIMESTEMP)
SRC = "/src/"

LOG_FILE_NAME = TIMESTEMP + ".log.txt"
TEMP_LOG_FILE_PATH = TEMP_DIR + LOG_FILE_NAME # start with temp dir
MAIN_FILE_NAME = "CNN_main.py"
COMMON_FILE_NAME = "common.py"


LOG_FILE_PATH = LOGS_DIR + "_" + LOG_FILE_NAME

#################################################
# NAMES
CONVOLUTION = "convolution"
MAX_POOL = "max_pool"
NORMALIZATION = "normalization"
FLATEN_4DTO2D = "flaten_4Dto2D"
DENSE = "dense"

FIT_INPUT = "fit_input"

NO_ACTIVATION = "NoActivaton"
RELU = "ReLU"
LEAKY_RELU = "Leaky_ReLU"

ON = "ON"
OFF = "OFF"

YES = "YES"
NO = "NO"

#################################################
# CONSTS
SEED = 1101

#################################################
# TOOLBOX
FRAC_RATIO_OFF = [OFF, 0, 0]
FRAC_RATIO_144 = [ON, 1.44, 1.44]
FRAC_RATIO_125 = [ON, 1.25, 1.25]
FRAC_RATIO_125_173 = [ON, 1.25, 1.73]
FRAC_RATIO_173_125 = [ON, 1.73, 1.25]
FRAC_RATIO_15_2 = [ON, 1.5, 2]

BIAS_CONST = 0.1
ALPHA_CONST = 0.2
WEIGHT_SDEV = 0.1
WEIGHT_MEAN = 0.003
INIT_VARS = [BIAS_CONST, ALPHA_CONST, WEIGHT_SDEV, WEIGHT_MEAN]

BS_80 = [80]
BS_100 = [100]
BS_1000 = [1000]

LR_003 = [0.003]
LR_001 = [0.001]
LR_0005 = [0.0005]
LR_0001 = [0.0001]
#################################################
# FLAGS



