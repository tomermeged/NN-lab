#################################################
# CIFAR-10 helper functions
#################################################
# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, 
# with 6000 images per class. There are 50000 training images and 10000 test images.

# The dataset is divided into five training batches and one test batch, each with 10000 images. 
# The test batch contains exactly 1000 randomly-selected images from each class. 
# The training batches contain the remaining images in random order, but some training 
# batches may contain more images from one class than another. Between them, 
# the training batches contain exactly 5000 images from each class.
#################################################


import common as CM
import numpy as np
import random

from utilities import print_and_log
from utilities import print_and_log_timestamp
from utilities import one_hot_encode

import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict

    
dirs = ['batches.meta', 'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
all_data = [0, 1, 2, 3, 4, 5, 6]
for i, direc in zip(all_data, dirs):
    all_data[i] = unpickle(CM.CIFAR_DIR + direc)

batch_meta = all_data[0]
data_batch1 = all_data[1]
data_batch2 = all_data[2]
data_batch3 = all_data[3]
data_batch4 = all_data[4]
data_batch5 = all_data[5]
test_batch = all_data[6]

num_labels = len(batch_meta[b'label_names'])
# print_and_log("{} categories: {}", num_labels, batch_meta[b'label_names'])

# Loaded in this way, each of the batch files contains a dictionary with the following elements:
# * data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image.
# The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue.
# The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
# * labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the i-th image in the array data.


####################################################################
# EXAMPLE HOW TO DIISPLAY IMAGES FROM CIFAR-10
# X_single = data_batch1[b'data'][1]
# X_single = X_single.reshape(3, 32, 32)
# X_single = X_single.transpose(1, 2, 0)  # reordering indexes [0,1,2 ] ==> [1,2,0]
#                                         # 0 = channel, 1 = row, 2 = col
# X_single = X_single.astype("uint8")
# plt.imshow(X_single)#
# plt.show()
#####################################################################

class CifarC:

    def __init__(self):
        self.train_index = 0

        self.all_train_batches = [data_batch1, data_batch2, data_batch3, data_batch4, data_batch5]
        self.test_batch = [test_batch]

        self.image_width = 32
        self.image_height = 32
        self.color_deapth = 255
        self.cs_rgb = 3

        self.training_images = None
        self.training_labels = None
        self.training_len = None
        
        self.training_images_shuffled = None
        self.training_labels_shuffled = None

        self.test_images = None
        self.test_labels = None
        self.test_len = None

    def set_up_images(self):
        """
        preparing the images for later use
        """
        print_and_log("Setting Up Training Images and Labels")

        # Vertically stacks the training images
        self.training_images = np.vstack([d[b"data"] for d in self.all_train_batches])
        self.training_len = len(self.training_images)

        # Reshapes and normalizes training images (x/255 is normalizing the colorspace)
        self.training_images = self.training_images.reshape(self.training_len, self.cs_rgb, self.image_width, self.image_height).transpose(0, 2, 3, 1) / self.color_deapth
        # One hot Encodes the training labels (e.g. [0,0,0,1,0,0,0,0,0,0])
        self.training_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.all_train_batches]), num_labels)

        print_and_log("Setting Up Test Images and Labels")
        
        self.training_images_shuffled = self.training_images
        self.training_labels_shuffled = self.training_labels
        
        random.Random(CM.SEED).shuffle(self.training_images_shuffled)
        random.Random(CM.SEED).shuffle(self.training_labels_shuffled)

        # Vertically stacks the test images
        self.test_images = np.vstack([d[b"data"] for d in self.test_batch])
        self.test_len = len(self.test_images)

        # Reshapes and normalizes test images
        self.test_images = self.test_images.reshape(self.test_len, self.cs_rgb, self.image_width, self.image_height).transpose(0, 2, 3, 1) / self.color_deapth
        # we want the shape to be [images, W, H, Channels]
        # One hot Encodes the test labels (e.g. [0,0,0,1,0,0,0,0,0,0])
        self.test_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.test_batch]), num_labels)

    def next_batch_train(self, batch_size):
        """
        return batch of images (x) and lables (y) according to batch_size
        """
        if self.training_len - self.train_index < batch_size:
            batch_size = self.training_len - self.train_index
            # print_and_log_timestamp("end of set, using batch_size {}", batch_size)
        x = self.training_images[self.train_index:self.train_index + batch_size].reshape(batch_size, self.image_width, self.image_height, self.cs_rgb)
        y = self.training_labels[self.train_index:self.train_index + batch_size]
        self.train_index = (self.train_index + batch_size) % self.training_len
        return x, y
        
    def next_batch_train_overlap(self, batch_size, overlap = 0):
        """
        return batch of images (x) and lables (y) according to batch_size with overlap between batches
        """
        if self.training_len - self.train_index <= batch_size - overlap:
            batch_size = self.training_len - self.train_index
            # print_and_log_timestamp("end of set, using batch_size {}", batch_size)
        if self.train_index - overlap < 0: overlap = 0
        x = self.training_images[self.train_index - overlap:self.train_index + (batch_size - overlap)].reshape(batch_size, self.image_width, self.image_height, self.cs_rgb)
        y = self.training_labels[self.train_index - overlap:self.train_index + (batch_size - overlap)]
        self.train_index = (self.train_index + batch_size - overlap) % self.training_len
        return x, y
