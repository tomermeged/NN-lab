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
MODEL_DIR = SAVED_MODELS_DIR + "Model_{}".format(TIMESTEMP)
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

CNNMODEL = "CNNmodel"

FIT_INPUT = "fit_input"

NO_ACTIVATION = "NoActivaton"
RELU = "ReLU"
LEAKY_RELU = "Leaky_ReLU"
ELU = "ELU"

ON = "ON"
OFF = "OFF"

YES = "YES"
NO = "NO"

#################################################
# CONSTS
SEED = 1101

#################################################
# TOOLBOX
STRIDE_1 = [1, 1]
STRIDE_2 = [2, 2]
STRIDE_3 = [3, 3]
STRIDE_4 = [4, 4]
STRIDE_15_2 = [1.5, 2]
STRIDE_144 = [1.44, 1.44]
STRIDE_125 = [1.25, 1.25]
STRIDE_125_173 = [1.25, 1.73]
STRIDE_173_125 = [1.73, 1.25]

BIAS_CONST = 0.1
ALPHA_CONST = 0.2
WEIGHT_SDEV = 0.1
WEIGHT_MEAN = 0.003
INIT_VARS = [BIAS_CONST, ALPHA_CONST, WEIGHT_SDEV, WEIGHT_MEAN]

BS_80 = [80]
BS_100 = [100]
BS_128 = [128]
BS_1000 = [1000]
BS_1200 = [1200]
BS_1500 = [1500]

LR_25 = [0.25]
LR_1 = [0.1]
LR_05 = [0.05]
LR_01 = [0.01]
LR_003 = [0.003]
LR_001 = [0.001]
LR_0005 = [0.0005]
LR_0001 = [0.0001]
LR_00001 = [0.00001]
#################################################
# FLAGS



