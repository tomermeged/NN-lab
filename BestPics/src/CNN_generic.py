#################################################
# CNN according to the LeCun architecture
#################################################

import common as CM
import tensorflow as tf

from utilities import print_and_log
from utilities import print_and_log_timestamp
from utilities import calc_params

from nn_components import inputLayer
from nn_components import convolutionLayer
from nn_components import denseLayer
from nn_components import maxPoolLayer
from nn_components import flaten4Dto2DLayer
from nn_components import normalizationLayer


#################################################
class genericCNN:

    def __init__(self, input):
        """
        initializing a cnn object with the first layer (input)
        """
        self.inputT = input
        self.NNlayer = []
        self.layer_ordinal = 0
        self.last_layer = None

        self.activation = CM.RELU
        self.kernelSize = 2
        self.stride = 1
        self.varInit = [0.1, 0.2, 0.1, 0.0]
        self.frac_ratio = CM.FRAC_RATIO_OFF
        self.dropout = CM.OFF


        with tf.name_scope("input"):
            layerObj = inputLayer(self.inputT)
            self.NNlayer.append([layerObj, layerObj.create_layer()])

    def override_defaults(self, activation=None, kernelSize=None, stride=None, varInit=None, frac_ratio=None, dropout=None):
        """
        will set new defaults for the following params:
        :activation: "no_activation", "relu", "leaky_relu" ; default = "relu"
        :kernelSize: for 2D layers, the kernel size (int) ; default = 2
        :stride:     for 2D layers, the stride size (int ; default = 2
        :varInit:    [bias_const, alpha_const, weight_sdev, weight_mean] ; default = [0.1, 0.2, 0.1, 0.0]
        :frac_ratio: [ON/OFF, col_ratio, row_ratio] ; default = "off"
        :dropout:    OFF/ dropout_value ; default = "off"
        """
        if activation != None:  self.activation = activation
        if kernelSize != None:  self.kernelSize = kernelSize
        if stride != None:      self.stride = stride
        if varInit != None:     self.varInit = varInit
        if frac_ratio != None:  self.frac_ratio = frac_ratio
        if dropout != None:     self.dropout = dropout

    def add_layer(self, type, layerSize, activation=None, kernelSize=None, stride=None, varInit=None, frac_ratio=None, dropout=None):
        """
        will add a layer to the model
        :types:      "fit_input", "convolution", "max_pool", "normalization", "flaten_4dto2d", "dense"
        :layerSize:  size of the layer (int)
        :activation: "no_activation", "relu", "leaky_relu" ; default = "relu"
        :kernelSize: for 2D layers, the kernel size (int) ; default = 2
        :stride:     for 2D layers, the stride size (int ; default = 2
        :varInit:    [bias_const, alpha_const, weight_sdev, weight_mean] ; default = [0.1, 0.2, 0.1, 0.0]
        :frac_ratio: [ON/OFF, col_ratio, row_ratio] ; default = "off"
        :dropout:    OFF/ dropout_value ; default = "off"
        """

        if activation == None:  activation = self.activation
        if kernelSize == None:  kernelSize = self.kernelSize
        if stride == None:      stride = self.stride
        if varInit == None:     varInit = self.varInit
        if frac_ratio == None:  frac_ratio = self.frac_ratio
        if dropout == None:     dropout = self.dropout

        inputT=self.NNlayer[self.layer_ordinal][1]

        if layerSize == CM.FIT_INPUT:
            layerSize = self.NNlayer[self.layer_ordinal][1]

        if type == CM.CONVOLUTION:
            with tf.name_scope("convolution_{}".format(self.layer_ordinal)):
                layerObj = convolutionLayer(inputT, layerSize, activation, kernelSize, stride, varInit, dropout)

        elif type == CM.MAX_POOL:
            with tf.name_scope("max_pool_{}".format(self.layer_ordinal)):
                layerObj = maxPoolLayer(inputT, layerSize, kernelSize, stride, frac_ratio)

        elif type == CM.NORMALIZATION:
            with tf.name_scope("Noramalization_{}".format(self.layer_ordinal)):
                layerObj = normalizationLayer(inputT)

        elif type == CM.FLATEN_4DTO2D:
            with tf.name_scope("flaten_4Dto2D"):
                layerObj = flaten4Dto2DLayer(inputT)

        elif type == CM.DENSE:
            with tf.name_scope("dense_{}".format(self.layer_ordinal)):
                layerObj = denseLayer(inputT, layerSize, activation, varInit, dropout)

        else:
            print_and_log("ERROR: unknown layer type {}", type)
            exit(1)

        output = layerObj.create_layer()
        self.NNlayer.append([layerObj, output])
        self.layer_ordinal += 1

    def print_model_params(self):
        """
        print model layers according to predefined layout
        """
        network_size = calc_params()
        print_and_log("==================================================================================")
        print_and_log("#    type             shape		        activation		dropout     additional")
        print_and_log("------------------+-------------------+----------------+-----------+-------------------")
        for layer in range(0, self.layer_ordinal + 1):
            self.NNlayer[layer][0].print_layer(layer)
        print_and_log("==================================================================================")
        print_and_log("Network Size:    {}", network_size)
        print_and_log("==================================================================================")


