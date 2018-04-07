#################################################
# Utilities functions
#################################################

import common
from common import *


# print_and_log
def print_and_log(str, var1=None, var2=None, var3=None, var4=None, var5=None):
    """
    used to print msgs to both screen and log file
    """
    print(str.format(var1, var2, var3, var4, var5))
    print(str.format(var1, var2, var3, var4, var5), file = open(TEMP_LOG_FILE_PATH, "a"))

def print_and_log_timestamp(str, var1=None, var2=None, var3=None, var4=None, var5=None):
    """
    used to print msgs to both screen and log file with a relative timestamp
    """
    now = time.time()
    time_diff = datetime.timedelta(seconds=(now - START_TIME))
    print(time_diff, end = ' ')
    print(str.format(var1, var2, var3, var4, var5))
    print(time_diff, end = ' ', file = open(TEMP_LOG_FILE_PATH, "a"))
    print(str.format(var1, var2, var3, var4, var5), file = open(TEMP_LOG_FILE_PATH, "a"))
    
def print_timestamp(str, var1=None, var2=None, var3=None, var4=None, var5=None):
    """
    used to print msgs to both screen and log file with a relative timestamp
    """
    now = time.time()
    time_diff = datetime.timedelta(seconds=(now - START_TIME))
    print(time_diff, end = ' ')
    print(str.format(var1, var2, var3, var4, var5))

def save_model(saver, sess, arch):
    """
    save model, src and log and print a msg
    """
    model_dir = MODEL_DIR + arch + "/"
    model_path = model_dir + arch
    model_src_dir = model_dir + SRC + "/"
    print_and_log_timestamp("==================================================================================")
    print_and_log_timestamp("Saving model: {}", model_path)
    print_and_log_timestamp("==================================================================================")
    create_dir(model_dir)
    saver.save(sess, model_path)
    copyfile(TEMP_LOG_FILE_PATH, model_path + "_" + LOG_FILE_NAME)
    if not os.path.exists(model_src_dir):
        copytree(TEMP_DIR_SRC, model_src_dir)

def restore_model(saver, sess, model_path):
    """
    restore saved model and print a msg
    """
    print_and_log_timestamp("==================================================================================")
    print_and_log_timestamp("restoring model: {}", model_path)
    print_and_log_timestamp("==================================================================================")
    saver.restore(sess, model_path)

def calc_params():
    """
    calculates the sum of all variables multiplications
    """
    return np.sum([np.prod(var.shape) for var in tf.trainable_variables()]) 

# HELPER FUNCTIONS FOR DEALING WITH DATA.
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

# HELPER FUNCTIONS FOR DEALING WITH DATA.
def create_dir(dir_name, dir_name_str=""):
    """
    make dir only if dir doesn't already exist
    """
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        print_and_log_timestamp("creating {} Dir: {}", dir_name_str, dir_name)
    
# LOGGING
create_dir(LOGS_DIR)
create_dir(TEMP_DIR)
print_and_log("temp log file path: {}", TEMP_LOG_FILE_PATH)
print_and_log("real log file path: {}", LOG_FILE_PATH)
print_and_log("TensorBoard path: {}", TENSORBOARD_PATH)
copytree(SRC_DIR, TEMP_DIR_SRC)
