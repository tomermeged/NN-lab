#################################################
# NN components functions
#################################################

from common import *

from utilities import print_and_log
from utilities import print_and_log_timestamp

# CREATING THE MODEL

# INIT BIAS
def init_bias(shape, init_val=0.1):
    init_dist = tf.constant(init_val, shape=shape)
    return tf.Variable(init_dist)
# about Bias init:
# suggests setting CNN biases to 0, quoting CS231n Stanford course:
# Initializing the biases. It is possible and common to initialize the biases to be zero,
# since the asymmetry breaking is provided by the small random numbers in the weights.
# For ReLU non-linearities, some people like to use small constant value such as 0.01
# for all biases because this ensures that all ReLU units fire in the beginning and therefore
# obtain and propagate some gradient. However, it is not clear if this provides a consistent
# improvement (in fact some results seem to indicate that this performs worse) and it is more
# common to simply use 0 bias initialization.

# INIT ALPHA
def init_alpha(shape, init_val=0.2):
    init_dist = tf.constant(init_val, shape=shape)
    return tf.Variable(init_dist)


# INIT WEIGHTS
def init_weight(shape, stddev=0.1, mean=0.0):
    init_dist = tf.truncated_normal(shape=shape, stddev=stddev, mean=mean, seed=SEED)
    return tf.Variable(init_dist)

    
    
# CONV 2D
def conv2d(inputT, filterT, strideS=1):  # 'T' stands for tensor
    """
    calcultes 2d convolution on 4d tensors
    """
    return tf.nn.conv2d(inputT, filterT, strides=[1, strideS, strideS, 1], padding='SAME')


# MAX POOLING
def max_pool_2x2(inputT, strideS, frac_ratio=None, kernelS=2): # S=size; inputT shape: [images, W, H, Channels]
    """
    inputT = the input Tensor
    strideS = size of each dim of a square stride
    """
    if frac_ratio == None or frac_ratio[0] == OFF:
        outputT = tf.nn.max_pool(inputT, 
                                ksize=[1, kernelS, kernelS, 1], 
                                strides=[1, strideS, strideS, 1], 
                                padding='SAME')
    else:
        outputT, row_seq, col_seq = tf.nn.fractional_max_pool(inputT,
                                        pooling_ratio=[1.0, frac_ratio[1], frac_ratio[2], 1.0], 
                                        pseudo_random=True)
    return outputT

       
# CONVOLUTION LAYER
def convolution_layer(inputT, patchS, layerS, actFunc, strideS=1, varInit=[0.1, 0.2, 0.1]):  # S=size; inputT shape: [images, W, H, Channels]
    """
    inputT = the input Tensor
    PatchS = size of each dim of a square conv patch
    layerS = size of the actual layer
    """
    Channels = int(inputT.get_shape()[3])
    biasT = init_bias([layerS], init_val=varInit[0])
    alpha = init_alpha([layerS], init_val=varInit[1])
    filterT = init_weight([patchS, patchS, Channels, layerS], stddev=varInit[2], mean=varInit[3])
    if actFunc == LEAKY_RELU:
        return tf.nn.leaky_relu(conv2d(inputT, filterT, strideS=strideS) + biasT, alpha=alpha)
    elif actFunc == RELU:
        return tf.nn.relu(conv2d(inputT, filterT, strideS=strideS) + biasT)
    else:
        print_and_log("unrecognized actFunc {}", actFunc)
        return None


def dense_layer(inputT, layerS, actFunc, varInit=[0.1, 0.2, 0.1, 0.0]):  # inputT shape: [images, W*H*Channels]
    """
    inputT = the input Tensor
    layerS = size of the actual layer
    actFunc = choose activation function foe layer
    """

    inputS = int(inputT.get_shape()[1])
    biasT = init_bias([layerS], init_val=varInit[0])
    alpha = init_alpha([layerS], init_val=varInit[1])
    filterT = init_weight([inputS, layerS], stddev=varInit[2], mean=varInit[3])
        
    if actFunc == LEAKY_RELU:
        return tf.nn.leaky_relu(tf.matmul(inputT, filterT) + biasT, alpha=alpha)
    elif actFunc == NO_ACTIVATION:
        return tf.matmul(inputT, filterT) + biasT
    elif actFunc == RELU:
        return tf.nn.relu(tf.matmul(inputT, filterT) + biasT)
    else:
        print_and_log("unrecognized actFunc {}", actFunc)
        return None


def output_layer(inputT, layerS):  # inputT shape: [images, W*H*Channels]
    """
    inputT = the input Tensor
    layerS = size of the actual layer
    """
    return dense_layer(inputT, layerS, actFunc=NO_ACTIVATION)

# FLATEN 4DTO2D
def flaten_4Dto2D(inputT):  # excpect the shape: [images, W*H*Channels]
    """
    flatten a tensor of shape [dim0, dim1, dim2, dim3] to [dim0, dim1 * dim2 * dim3]
    """
    dim1 = int(inputT.get_shape()[1])
    dim2 = int(inputT.get_shape()[2])
    dim3 = int(inputT.get_shape()[3])
    return tf.reshape(inputT, [-1, dim1 * dim2 * dim3])       
        
        
        
        
        
        
        
        