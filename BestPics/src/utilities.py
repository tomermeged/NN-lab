#################################################
# Utilities functions
#################################################


import common as CM
import os, sys
import datetime
import time
from shutil import copyfile, copytree

import numpy as np
import tensorflow as tf


# PRINT FUNCTIONS
def print_and_log(str, var1=None, var2=None, var3=None, var4=None, var5=None, var6=None, var7=None, var8=None, var9=None):
    """
    used to print msgs to both screen and log file
    """
    print(str.format(var1, var2, var3, var4, var5, var6, var7, var8, var9))
    print(str.format(var1, var2, var3, var4, var5, var6, var7, var8, var9), file = open(CM.TEMP_LOG_FILE_PATH, "a"))

def print_and_log_timestamp(str, var1=None, var2=None, var3=None, var4=None, var5=None, var6=None, var7=None, var8=None, var9=None):
    """
    used to print msgs to both screen and log file with a relative timestamp
    """
    now = time.time()
    time_diff = datetime.timedelta(seconds=(now - CM.START_TIME))
    print(time_diff, end = ' ')
    print(str.format(var1, var2, var3, var4, var5, var6, var7, var8, var9))
    print(time_diff, end = ' ', file = open(CM.TEMP_LOG_FILE_PATH, "a"))
    print(str.format(var1, var2, var3, var4, var5, var6, var7, var8, var9), file = open(CM.TEMP_LOG_FILE_PATH, "a"))

def print_timestamp(str, var1=None, var2=None, var3=None, var4=None, var5=None, var6=None, var7=None, var8=None, var9=None):
    """
    used to print msgs to both screen and log file with a relative timestamp
    """
    now = time.time()
    time_diff = datetime.timedelta(seconds=(now - CM.START_TIME))
    print(time_diff, end = ' ')
    print(str.format(var1, var2, var3, var4, var5, var6, var7, var8, var9))


# CALCULATE PARAMS
def calc_params():
    """
    calculates the sum of all variables multiplications
    """
    return np.sum([np.prod(var.shape) for var in tf.trainable_variables()])

# ONE HOT ENCODING
def one_hot_encode(vec, vals):
    """
    one-hot encode for labels
    :param vec: vec are the images as vector (hstack)
    :param vals: number of values to encode for
    :return: one hot encoding for the vector
    """
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out

# CREATE DIR
def create_dir(dir_name, dir_name_str=""):
    """
    make dir only if dir doesn't already exist
    """
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        print_and_log_timestamp("creating {} Dir: {}", dir_name_str, dir_name)



# SAVE MODEL
def save_model(saver, sess):
    """
    save model, src and log and print a msg
    """
    model_dir = CM.MODEL_DIR + "/"
    model_path = model_dir + CM.CNNMODEL
    model_src_dir = model_dir + CM.SRC + "/"
    print_and_log_timestamp("==================================================================================")
    print_and_log_timestamp("Saving model: {}", model_path)
    print_and_log_timestamp("==================================================================================")
    create_dir(model_dir)
    saver.save(sess, model_path)
    copyfile(CM.TEMP_LOG_FILE_PATH, model_dir + "_" + CM.LOG_FILE_NAME)
    if not os.path.exists(model_src_dir):
        copytree(CM.TEMP_DIR_SRC, model_src_dir)

# RESTORE MODEL
def restore_model(saver, sess, model_name):
    """
    restore saved model and print a msg
    """
    model_path = CM.SAVED_MODELS_DIR + model_name + "/" + CM.CNNMODEL
    print_and_log_timestamp("==================================================================================")
    print_and_log_timestamp("restoring model: {}", model_path)
    print_and_log_timestamp("==================================================================================")
    saver.restore(sess, model_path)


# LOGGING
create_dir(CM.LOGS_DIR)
create_dir(CM.TEMP_DIR)
print_and_log("temp log file path: {}", CM.TEMP_LOG_FILE_PATH)
print_and_log("real log file path: {}", CM.LOG_FILE_PATH)
print_and_log("TensorBoard path: {}", CM.TENSORBOARD_PATH)
copytree(CM.SRC_DIR, CM.TEMP_DIR_SRC)
