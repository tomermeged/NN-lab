#################################################
# CNN MAIN ROUTINE
#################################################

import common as CM
import math

import datetime
import time
from shutil import copyfile, copytree

import tensorflow as tf

from utilities import print_and_log
from utilities import print_and_log_timestamp
from utilities import print_timestamp
from utilities import calc_params
from utilities import save_model
from utilities import restore_model

from cifar_10_handler import CifarC
from cifar_10_handler import num_labels

import CNN_generic


#################################################
# CONSTS
TEST_DROPOUT = 1.0
TEST_LR = 1
BEST_ACCURACY_THRESHOLD = 0.7
MIN_EPOCHS_TO_SAVE = 7
MIN_ACCURACY_DISCARD = 0.2
MIN_EPOCHS_DISCARD = 3
MIN_DROPOUT = 0.3
DROPOUT_UPDATE_MIN_GAP = 0.1
DROPOUT_UPDATE_FACTOR = 0.95

#################################################
# CONTROLS
TEST_ONLY = CM.OFF
TRAIN_AGAIN = CM.OFF
TRAINED_MODEL_NAME = "Model_20180512_100505" # when restoring a model

#################################################
# TRAINING PARAMS
OVERLAP = CM.OFF
MOMENTUM = CM.OFF
RUN_STEPS_FACTOR = 1
TRAIN_DROPOUT = 0.65
LR_PROGRESSION = 10 # in epochs
BATCH_SIZE = CM.BS_80 * 15
LEARNING_RATE = CM.LR_003 * 15
NUM_EPOCHS = len(BATCH_SIZE) * LR_PROGRESSION

if len(BATCH_SIZE) != len(LEARNING_RATE):
    exit(2)



#################################################
#################################################
# FUNCTIONS
def print_params():
    print_and_log("==================================================================================")
    print_and_log("seed:            {}", CM.SEED)
    print_and_log("==================================================================================")
    print_and_log("learning_rate    {} every {} epochs", LEARNING_RATE, LR_PROGRESSION)
    print_and_log("momentum         {}", MOMENTUM)
    print_and_log("train_dropout    {}", TRAIN_DROPOUT)


def test_model(sess, y_pred, y_true, feed_dict_test, writer=None, merged=None, epoch=None):

    matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
    acc = tf.reduce_mean(tf.cast(matches,tf.float32))

    
    result, summary = sess.run([acc, merged], feed_dict=feed_dict_test)
    writer.add_summary(summary, epoch)
    return result
#################################################

CIFR = CifarC()
CIFR.set_up_images()

# PLACEHOLDERS:
x = tf.placeholder(tf.float32, shape=[None, CIFR.image_width, CIFR.image_height, CIFR.cs_rgb], name='input')
y_true = tf.placeholder(tf.float32, shape=[None, num_labels], name='y_true')
dropout = tf.placeholder(tf.float32, name='dropout')
learning_rate = tf.placeholder(tf.float32, name='learning_rate')

print_params()

Cnn = CNN_generic.genericCNN(x, CIFR.cs_rgb)
##################################################################################################
# LAYERS
# add_layer args:
#   type, layerSize, activation, kernelSize, stride, varInit, frac_ratio, dropout
Cnn.override_defaults(activation = "ELU", 
                kernelSize = 2, 
                stride = [1, 1], 
                varInit = [0.1, 0.2, 0.1, 0.003], 
                dropout = "OFF")


Cnn.add_layer(CM.CONVOLUTION, 32, kernelSize = 4)
Cnn.add_layer(CM.MAX_POOL, kernelSize = 3, stride = CM.STRIDE_2)

Cnn.add_layer(CM.NORMALIZATION)
Cnn.add_layer(CM.CONVOLUTION, 64)
Cnn.add_layer(CM.MAX_POOL, stride = CM.STRIDE_144)

Cnn.add_layer(CM.NORMALIZATION)
Cnn.add_layer(CM.CONVOLUTION, 128, dropout = dropout)
Cnn.add_layer(CM.MAX_POOL, stride = CM.STRIDE_144)

Cnn.add_layer(CM.NORMALIZATION)
Cnn.add_layer(CM.CONVOLUTION, 256, dropout = dropout)
Cnn.add_layer(CM.MAX_POOL, stride = CM.STRIDE_144)

Cnn.add_layer(CM.NORMALIZATION)
Cnn.add_layer(CM.CONVOLUTION, 384, dropout = dropout)
Cnn.add_layer(CM.MAX_POOL, stride = CM.STRIDE_144)

Cnn.add_layer(CM.NORMALIZATION)
Cnn.add_layer(CM.CONVOLUTION, 512, dropout = dropout)
Cnn.add_layer(CM.MAX_POOL, stride = CM.STRIDE_144)

Cnn.add_layer(CM.FLATEN_4DTO2D, None)
Cnn.add_layer(CM.DENSE, num_labels, "NoActivaton")
##################################################################################################



Cnn.print_model_params()
# output Layer
with tf.name_scope("output_Layer"):
    y_pred = Cnn.NNlayer[Cnn.layer_ordinal].outputT

# SOFTMAX and LOSS FUNCTION
with tf.name_scope("loss"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)
    loss_func = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('loss', loss_func)
    # OPTIMIZER
    if MOMENTUM != CM.OFF:
        optimizer = tf.train.MomentumOptimizer(learning_rate, MOMENTUM)
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate)

    train = optimizer.minimize(loss_func)


matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
acc = tf.reduce_mean(tf.cast(matches,tf.float32))

with tf.name_scope("test_accuracy"):
    tf.summary.scalar('test_accuracy', acc)
    merged_test = tf.summary.merge_all()


# RUN PREPERATIONS
saver = tf.train.Saver() # so we can save the model
init_vars = tf.global_variables_initializer()
feed_dict_test={x: CIFR.test_images, y_true: CIFR.test_labels, dropout: TEST_DROPOUT, learning_rate: TEST_LR}
train_dropout = TRAIN_DROPOUT # initial dropout value
best_test_accuracy = BEST_ACCURACY_THRESHOLD
print_and_log("{} START session:", datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S"))

if TEST_ONLY == CM.ON:
    # TEST ALREADY TRAINED MODEL:
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(TCM.ENSORBOARD_PATH, sess.graph)
        # RESTORE MODEL
        restore_model(sess, TRAINED_MODEL_NAME) # supplying the same name - it's not a filename!!
        # TEST MODEL
        test_accuracy, test_summary = sess.run([acc, merged_test], feed_dict=feed_dict_test)
        writer.add_summary(test_summary, 0)
        print_and_log_timestamp("accuracy: {}", test_accuracy)
        writer.close()
else:
    # TRAINING A MODEL:
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(CM.TENSORBOARD_PATH, sess.graph)
        if TRAIN_AGAIN == CM.ON:
            # CONTINUE TRAINING OF AN EXISTING (PRE-TRAINED) MODEL:
            restore_model(saver, sess, TRAINED_MODEL_NAME) # supplying the same name - it's not a filename!!
        else:
            sess.run(init_vars)

        test_accuracy, test_summary = sess.run([acc, merged_test], feed_dict=feed_dict_test)
        writer.add_summary(test_summary, 0)
        print_and_log_timestamp("epoch {}/{}: test accuracy is {}", 0, NUM_EPOCHS, test_accuracy)

        for epoch in range(NUM_EPOCHS):
            if epoch > MIN_EPOCHS_TO_SAVE: copyfile(CM.TEMP_LOG_FILE_PATH, CM.LOG_FILE_PATH)

            batch_size = BATCH_SIZE[math.floor(epoch / LR_PROGRESSION)]
            lr = LEARNING_RATE[math.floor(epoch / LR_PROGRESSION)]
            if OVERLAP == CM.OFF:
                run_steps = math.ceil(CIFR.training_len / batch_size  * RUN_STEPS_FACTOR)
            else:
                run_steps = math.ceil(CIFR.training_len / (batch_size - OVERLAP) * RUN_STEPS_FACTOR)
            if run_steps == 0: run_steps = 1
            print_and_log_timestamp("epoch {}/{}: will run {} steps with batch_size {} ; lr={}", epoch+1, NUM_EPOCHS, run_steps, batch_size, lr)
            for step in range(run_steps):
                if OVERLAP == CM.OFF:
                    batch_x_train, batch_y_train = CIFR.next_batch_train(batch_size)
                else:
                    batch_x_train, batch_y_train = CIFR.next_batch_train_overlap(batch_size, OVERLAP)
                feed_dict_train={ x: batch_x_train, y_true: batch_y_train, dropout: train_dropout, learning_rate: lr}
                sess.run(train, feed_dict=feed_dict_train)

                if step % 50 == 0: print_timestamp("step {}", step)

            train_accuracy = sess.run(acc, feed_dict=feed_dict_train)
            # train_accuracy, summary = sess.run([acc_1, merged_train], feed_dict=feed_dict_train)
            # writer.add_summary(summary, epoch)

            print_and_log_timestamp("epoch {}/{}: train accuracy is {}", epoch+1, NUM_EPOCHS, train_accuracy)
            if train_accuracy < MIN_ACCURACY_DISCARD and epoch >= MIN_EPOCHS_DISCARD:
                print_and_log_timestamp("model is not learning...exit")
                exit(1)
            if train_accuracy > best_test_accuracy:
                test_accuracy, test_summary = sess.run([acc, merged_test], feed_dict=feed_dict_test)
                writer.add_summary(test_summary, epoch)
                print_and_log_timestamp("epoch {}/{}: test accuracy is {}", epoch+1, NUM_EPOCHS, test_accuracy)
                if test_accuracy > best_test_accuracy:
                    best_test_accuracy = test_accuracy # updating best test accuracy
                    save_model(saver, sess) # save model
                if train_accuracy - test_accuracy > DROPOUT_UPDATE_MIN_GAP and train_dropout > MIN_DROPOUT:
                    train_dropout = train_dropout * DROPOUT_UPDATE_FACTOR # updating dropout
                    print_and_log_timestamp("**** updating dropout value to {} ****", train_dropout)


        writer.close()
    print_and_log('\n')

print_and_log_timestamp(" END session! {}", datetime.datetime.now().strftime("%H:%M:%S"))
copyfile(CM.TEMP_LOG_FILE_PATH, CM.LOG_FILE_PATH)




