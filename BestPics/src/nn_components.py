#################################################
# NN components functions
#################################################

import common as CM
import tensorflow as tf

from utilities import print_and_log
from utilities import print_and_log_timestamp

# CREATING THE MODEL

# INIT BIAS
def init_bias(shape, init_val=0.1):
    init_dist = tf.constant(init_val, shape=shape)
    bias = tf.Variable(init_dist, name='bias')
    tf.summary.histogram('bias_hist', bias)
    return bias
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
    alpha = tf.Variable(init_dist, name='alpha')
    tf.summary.histogram('alpha_hist', alpha)
    return alpha


# INIT WEIGHTS
def init_weight(shape, stddev=0.1, mean=0.0):
    init_dist = tf.truncated_normal(shape=shape, stddev=stddev, mean=mean, seed=CM.SEED)
    weight = tf.Variable(init_dist, name='weight')
    tf.summary.histogram('weight_hist', weight)
    return weight



# CONV 2D
def conv2d(inputT, filterT, stride=1):  # 'T' stands for tensor
    """
    calcultes 2d convolution on 4d tensors
    :inputT: the input Tensor
    :filterT: the filter Tensor
    :stride: size of each dim of a square stride
    """
    return tf.nn.conv2d(inputT, filterT, strides=[1, stride[0], stride[1], 1], padding='SAME')


class layerProp():

    def __init__(self, inputT, layerSize, activation=CM.RELU, kernelSize=2, stride=1, varInit=[0.1, 0.2, 0.1, 0.0], frac_ratio=None):
        return None


class Genlayer():  # S=size; inputT shape: [images, W, H, Channels]

    def __init__(self, inputT, layerSize, varInit):
        return None

class inputLayer():  # S=size; inputT shape: [images, W, H, Channels]

    def __init__(self, inputT):
        self.type = "input"
        self.inputT = inputT
        self.outputT = None

    def create_layer(self):
        self.outputT = self.inputT
        return self.outputT

    def print_layer(self, layer_ordinal):
        print_and_log("{}\t{}\t\t\t{}", layer_ordinal, self.type, self.outputT.shape)


class convolutionLayer(Genlayer):  # S=size; inputT shape: [images, W, H, Channels]

    def __init__(self, inputT, layerSize, activation=CM.RELU, kernelSize=2, stride=1, varInit=[0.1, 0.2, 0.1, 0.0], dropout=CM.OFF):
        Genlayer.__init__(self, inputT, layerSize, varInit)
        self.type = "convolution"
        self.inputT = inputT
        self.layerSize = layerSize
        self.activation = activation
        self.kernelSize = kernelSize
        self.stride = stride
        self.bias_const = varInit[0]
        self.alpha_const = varInit[1]
        self.weight_sdev = varInit[2]
        self.weight_mean = varInit[3]
        self.dropout = dropout
        if self.dropout == CM.OFF:
            self.dropout_str = CM.OFF
        else:
            self.dropout_str = CM.ON
        self.outputT = None


    def create_layer(self):
        biasT = init_bias([self.layerSize], self.bias_const)
        alpha = init_alpha([self.layerSize], self.alpha_const)
        Channels = int(self.inputT.get_shape()[3])
        filterT = init_weight([self.kernelSize, self.kernelSize, Channels, self.layerSize], self.weight_sdev, self.weight_mean)
        if self.activation == CM.ELU:
            self.outputT = tf.nn.elu(conv2d(self.inputT, filterT, self.stride) + biasT)        
        elif self.activation == CM.LEAKY_RELU:
            self.outputT = tf.nn.leaky_relu(conv2d(self.inputT, filterT, self.stride) + biasT, alpha)
        elif self.activation == CM.RELU:
            self.outputT = tf.nn.relu(conv2d(self.inputT, filterT, self.stride) + biasT)
            self.alpha_const = "NA"
        elif self.activation == CM.NO_ACTIVATION:
            self.outputT = conv2d(self.inputT, filterT, self.stride) + biasT
            self.alpha_const = "NA"
        else:
            print_and_log("ERROR: unknown activation {}", self.activation)
            exit(1)

        if self.dropout == CM.OFF:
            return self.outputT
        else:
            return tf.nn.dropout(self.outputT, self.dropout)


    def print_layer(self, layer_ordinal):
        print_and_log("{}\t{}\t\t{}\t\t{}\t\t\t{}\t\t\tK={} ; S={}", layer_ordinal, self.type, self.outputT.shape, self.activation, self.dropout_str, self.kernelSize, self.stride)


class denseLayer():  # S=size; inputT shape: [images, W, H, Channels]

    def __init__(self, inputT, layerSize, activation=CM.RELU, varInit=[0.1, 0.2, 0.1, 0.0], dropout=CM.OFF):
        self.type = "dense"
        self.inputT = inputT
        self.layerSize = layerSize
        self.activation = activation
        self.bias_const = varInit[0]
        self.alpha_const = varInit[1]
        self.weight_sdev = varInit[2]
        self.weight_mean = varInit[3]
        self.dropout = dropout
        if self.dropout == CM.OFF:
            self.dropout_str = CM.OFF
        else:
            self.dropout_str = CM.ON
        self.outputT = None



    def create_layer(self):
        biasT = init_bias([self.layerSize], self.bias_const)
        alpha = init_alpha([self.layerSize], self.alpha_const)
        inputSize = int(self.inputT.get_shape()[1])
        filterT = init_weight([inputSize, self.layerSize], self.weight_sdev, self.weight_mean)
        if self.activation == CM.ELU:
            self.outputT = tf.nn.elu(tf.matmul(self.inputT, filterT) + biasT, alpha)
        elif self.activation == CM.LEAKY_RELU:
            self.outputT = tf.nn.leaky_relu(tf.matmul(self.inputT, filterT) + biasT, alpha)
        elif self.activation == CM.RELU:
            self.outputT = tf.nn.relu(tf.matmul(self.inputT, filterT) + biasT)
            self.alpha_const = "NA"
        elif self.activation == CM.NO_ACTIVATION:
            self.outputT = tf.matmul(self.inputT, filterT) + biasT
            self.alpha_const = "NA"
        else:
            print_and_log("ERROR: unknown activation {}", self.activation)
            exit(1)

        if self.dropout == CM.OFF:
            return self.outputT
        else:
            return tf.nn.dropout(self.outputT, self.dropout)


    def print_layer(self, layer_ordinal):
        print_and_log("{}\t{}\t\t\t{}\t\t{}\t\t{}", layer_ordinal, self.type, self.outputT.shape, self.activation, self.dropout_str)


class maxPoolLayer():  # S=size; inputT shape: [images, W, H, Channels]

    def __init__(self, inputT, kernelSize=2, stride=[2, 2], frac_ratio=None):
        self.type = "max_pool"
        self.inputT = inputT
        self.kernelSize = kernelSize
        self.stride = stride
        self.frac_ratio = frac_ratio
        self.outputT = None


    def create_layer(self):
        # if self.frac_ratio == None or self.frac_ratio[0] == CM.OFF:
            # self.outputT = tf.nn.max_pool(self.inputT,
                                    # ksize=[1, self.kernelSize, self.kernelSize, 1],
                                    # strides=[1, self.stride, self.stride, 1],
                                    # padding='SAME')
        # else:
        self.outputT, _, _ = tf.nn.fractional_max_pool(self.inputT,
                                            pooling_ratio=[1.0, self.stride[0], self.stride[1], 1.0],
                                            pseudo_random=True)
        return self.outputT


    def print_layer(self, layer_ordinal):
        if self.frac_ratio == None or self.frac_ratio[0] == CM.OFF:
            print_and_log("{}\t{}\t\t{}\t\t\t\t\t\t\t\t\tK={} ; S={}", layer_ordinal, self.type, self.outputT.shape, self.kernelSize, self.stride)
        else:
            print_and_log("{}\t{}\t\t{}\t\t\t\t\t\t\t\t\tK={} ; S={} frac_ratio={}", layer_ordinal, self.type, self.outputT.shape, self.kernelSize, self.stride, self.frac_ratio)


class normalizationLayer():  # S=size; inputT shape: [images, W, H, Channels]

    def __init__(self, inputT):
        self.type = "normalization"
        self.inputT = inputT
        self.outputT = None


    def create_layer(self):
        """
        normalization layer
        """
        self.outputT = tf.contrib.layers.layer_norm(self.inputT)
        return self.outputT

    def print_layer(self, layer_ordinal):
        print_and_log("{}\t{}\t{}", layer_ordinal, self.type, self.outputT.shape)

class flaten4Dto2DLayer():  # S=size; inputT shape: [images, W, H, Channels]

    def __init__(self, inputT):
        self.type = "flaten4Dto2D"
        self.inputT = inputT
        self.outputT = None


    def create_layer(self):
        """
        flatten a tensor of shape [dim0, dim1, dim2, dim3] to [dim0, dim1 * dim2 * dim3]
        """
        dim1 = int(self.inputT.get_shape()[1])
        dim2 = int(self.inputT.get_shape()[2])
        dim3 = int(self.inputT.get_shape()[3])
        self.outputT = tf.reshape(self.inputT, [-1, dim1 * dim2 * dim3])
        return self.outputT

    def print_layer(self, layer_ordinal):
        print_and_log("{}\t{}\t{}", layer_ordinal, self.type, self.outputT.shape)

