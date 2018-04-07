#################################################
# CNN MAIN ROUTINE
#################################################


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

from common import *
import math

from utilities import print_and_log
from utilities import print_and_log_timestamp
from utilities import print_timestamp
from utilities import calc_params
from utilities import restore_model
from utilities import save_model

from cifar_10_handler import CifarC
from cifar_10_handler import num_labels

from nn_components import output_layer

import CNN_lecun
import CNN_lecunBoost
import CNN_NIN
import CNN_hertel_ijcnn

 
def test_model(sess, y_pred, y_true, feed_dict_test):
    matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
    acc = tf.reduce_mean(tf.cast(matches,tf.float32))
    result = sess.run(acc, feed_dict=feed_dict_test)
    return result

def print_params():
    print_and_log("==================================================================================")
    print_and_log("{} Architecture", ARCH)
    print_and_log("seed:            {}", SEED)
    print_and_log("==================================================================================")
    print_and_log("learning_rate    {} every {} epochs", LEARNING_RATE, LEARNING_RATE_PROGRESSION)
    print_and_log("momentum         {}", MOMENTUM)
    print_and_log("train_dropout    {}", TRAIN_DROPOUT)


CIFR = CifarC()
CIFR.set_up_images()
    
# PLACEHOLDERS:
x = tf.placeholder(tf.float32, shape=[None, CIFAR_IMAGE_WIDTH, CIFAR_IMAGE_HEIGHT, CS_RGB])
y_true = tf.placeholder(tf.float32, shape=[None, num_labels])
dropout = tf.placeholder(tf.float32)
learning_rate = tf.placeholder(tf.float32)


        
        

print_params()
    
if ARCH == Lecun:
    Cnn = CNN_lecun.CnnLecun()
elif ARCH == LecunBoost:
    Cnn = CNN_lecunBoost.CnnLecunBoost()
elif ARCH == NIN:
    Cnn = CNN_NIN.CnnNin()
elif ARCH == hertel_ijcnn:
    Cnn = CNN_hertel_ijcnn.hertel_ijcnn()
else:
    print_and_log("No such Architecture found: '{}'", ARCH)
    exit(1)

# output Layer
output = Cnn.create_network(x, y_true, dropout)
y_pred = output_layer(output, num_labels)
print_and_log("==================================================================================")
print_and_log("output:    {}", output.shape)        
print_and_log("softmax:    {}", y_pred.shape)        
print_and_log("==================================================================================")
# SOFTMAX and LOSS FUNCTION
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)
loss_func = tf.reduce_mean(cross_entropy)

# OPTIMIZER
if MOMENTUM != OFF:
    optimizer = tf.train.MomentumOptimizer(learning_rate, MOMENTUM)
else:
    optimizer = tf.train.AdamOptimizer(learning_rate)
    
train = optimizer.minimize(loss_func)
        

# SAVER
saver = tf.train.Saver() # so we can save the model

# RUN
init_vars = tf.global_variables_initializer()
# feed_dict_test_mini={x: CIFR.test_images[:TEST_BATCH_SIZE], y_true: CIFR.test_labels[:TEST_BATCH_SIZE], dropout: TEST_DROPOUT, learning_rate: 1}
feed_dict_test={x: CIFR.test_images, y_true: CIFR.test_labels, dropout: TEST_DROPOUT, learning_rate: 1}

print_and_log("{} START session:", datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S"))
best_test_accuracy = ACC_THRESHOLD
if TEST_ONLY == ON:
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(TENSORBOARD_PATH, sess.graph)
        # RESTORE MODEL
        restore_model(sess, TRAINED_MODEL) # supplying the same name - it's not a filename!!
        # TEST MODEL
        test_accuracy = test_model(sess, y_pred, y_true, feed_dict_test)
        print_and_log_timestamp("accuracy: {}", test_accuracy)
        writer.close()
else:
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(TENSORBOARD_PATH, sess.graph)
        if TRAIN_AGAIN == ON:
            restore_model(saver, sess, TRAINED_MODEL) # supplying the same name - it's not a filename!!
        else:
            sess.run(init_vars)
                   
        for epoch in range(NUM_EPOCHS):
            if epoch > EPOCH_THRESHOLD:
                copyfile(TEMP_LOG_FILE_PATH, LOG_FILE_PATH)
            batch_size=BATCH_SIZE[epoch]            
            run_steps = math.ceil(CIFAR_TRAIN_SIZE / batch_size * RUN_STEPS_FACTOR)
            if run_steps == 0: run_steps = 1
            print_and_log_timestamp("epoch {}/{}: will run {} steps with batch_size {}", epoch+1, NUM_EPOCHS, run_steps, batch_size)
            for step in range(run_steps):
                batch_x_train, batch_y_train = CIFR.next_batch_train(batch_size)
                feed_dict_train={ x: batch_x_train, y_true: batch_y_train, dropout: TRAIN_DROPOUT, learning_rate: LEARNING_RATE[math.floor(epoch / LEARNING_RATE_PROGRESSION)]}    
                sess.run(train, feed_dict=feed_dict_train)
                
                if step % 50 == 0:
                    print_timestamp("step {}", step)
                            
            train_accuracy = test_model(sess, y_pred, y_true, feed_dict_train)
            # test_accuracy = test_model(sess, y_pred, y_true, feed_dict_test_mini)
            print_and_log_timestamp("epoch {}/{}: train accuracy is {}", epoch+1, NUM_EPOCHS, train_accuracy)
            if train_accuracy > best_test_accuracy:
                test_accuracy = test_model(sess, y_pred, y_true, feed_dict_test)
                print_and_log_timestamp("epoch {}/{}: test accuracy is {}", epoch+1, NUM_EPOCHS, test_accuracy)            
                if test_accuracy > best_test_accuracy:
                    best_test_accuracy = test_accuracy
                    save_model(saver, sess, ARCH)
        
        writer.close()
    print_and_log('\n')

print_and_log_timestamp(" END session! {}", datetime.datetime.now().strftime("%H:%M:%S"))
copyfile(TEMP_LOG_FILE_PATH, LOG_FILE_PATH)




