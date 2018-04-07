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


import os, sys
import datetime
import time
from shutil import copyfile, copytree

import numpy as np
import tensorflow as tf

#################################################
# DEFS
CIFAR_DIR = '../cifar-10-batches-py/'
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

START_TIME = time.time()

Lecun = "Lecun"
LecunBoost = "LecunBoost"
NIN = "NIN"
hertel_ijcnn = "hertel_ijcnn"

ACC_THRESHOLD = 0.7
EPOCH_THRESHOLD = 7
#################################################
# CONSTS
SEED = 1101
CIFAR_IMAGE_WIDTH = 32
CIFAR_IMAGE_HEIGHT = 32
CS_RGB = 3
COLOR_DEAPTH = 255

CIFAR_TRAIN_SIZE = 50000 #50000
CIFAR_TEST_SIZE = 10000 #10000

NO_ACTIVATION = "NoActivaton"
RELU = "ReLU"
LEAKY_RELU = "Leaky_ReLU"

TEST_DROPOUT = 1.0

ON = "ON"
OFF = "OFF"

YES = "YES"
NO = "NO"
#################################################
# FLAGS

#################################################
# CONTROLS
TEST_ONLY = OFF
TRAIN_AGAIN = OFF

ARCH = hertel_ijcnn # Lecun, LecunBoost, NIN, hertel_ijcnn

TRAINED_MODEL_DIR = "Model_20180407_044224_hertel_ijcnn"
TRAINED_MODEL = SAVED_MODELS_DIR + TRAINED_MODEL_DIR + "/" + ARCH
LOG_FILE_PATH = LOGS_DIR + ARCH + "_" + LOG_FILE_NAME

#################################################
# GENERAL MODEL PARAMS

BATCH_SIZE = [  100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
                1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
                1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
                
# BATCH_SIZE = [  1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
                # 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
                
             # 50, 100, 200, 500, 1000, 2000
NUM_EPOCHS = len(BATCH_SIZE)
RUN_STEPS_FACTOR = 1

LEARNING_RATE = [0.003, 0.001, 0.0005, 0.0001, 0.0001, 0.0001]
# LEARNING_RATE = [0.0001, 0.0001, 0.0001, 0.0001]
LEARNING_RATE_PROGRESSION = 10 # in epochs
MOMENTUM = OFF
TRAIN_DROPOUT = 0.6
#################################################

