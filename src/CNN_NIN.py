#################################################
# CNN according to the LeCun architecture
#################################################

from common import *

from utilities import print_and_log
from utilities import print_and_log_timestamp
from utilities import calc_params

from nn_components import max_pool_2x2
from nn_components import convolution_layer
from nn_components import dense_layer
from nn_components import flaten_4Dto2D


#################################################
# MODEL SPECIFIC PARAMS
CONV_LAYERS = 4
CONV_LAYER_SIZE = [192, 192, 192, 64, 64, 64]
CONV_PATCH_SIZE = 4
CONV_ACTIVATION = LEAKY_RELU

MAX_POOL_STRIDE = 2
MAX_POOL_FRAC_RATIO = [OFF, 1.44, 1.44]

DENSE_LAYERS = 0
DENSE_LAYERS_SIZE = [812, 812]
DENSE_ACTIVATION = LEAKY_RELU

BIAS_CONST = 0.1
ALPHA_CONST = 0.2
WEIGHT_SDEV = 0.1


#################################################
      
class CnnNin:

    def __init__(self):
        self.input = None
        self.NNlayer = []
        self.layer_ordinal = 0
        
        self.conv_layers = CONV_LAYERS
        self.conv_layer_size = CONV_LAYER_SIZE
        self.conv_patch_size = CONV_PATCH_SIZE
        self.conv_activation = CONV_ACTIVATION
        self.max_pool_stride = MAX_POOL_STRIDE
        self.max_pool_frac_ratio = MAX_POOL_FRAC_RATIO
        
        self.dense_layers = DENSE_LAYERS
        self.dense_layers_size = DENSE_LAYERS_SIZE
        self.dense_activation = DENSE_ACTIVATION
        self.bias_const = BIAS_CONST
        self.alpha_const = ALPHA_CONST
        self.weight_sdev = WEIGHT_SDEV
        

    def print_model_params(self, x, y_true, dropout):
        network_size = calc_params()
        print_and_log("==================================================================================")
        print_and_log("conv_layers          {}", self.conv_layers)
        print_and_log("conv_layer_size		{}", self.conv_layer_size)
        print_and_log("conv_patch_size		{}", self.conv_patch_size)
        print_and_log("conv_activation		{}", self.conv_activation)
        print_and_log("max_pool_stride		{}", self.max_pool_stride)
        print_and_log("max_pool_frac_ratio  {}", self.max_pool_frac_ratio)

        print_and_log("dense_layers         {}", self.dense_layers)
        print_and_log("dense_layers_size    {}", self.dense_layers_size)
        print_and_log("dense_activation		{}", self.dense_activation)
        print_and_log("bias_const           {}", self.bias_const)
        print_and_log("alpha_const          {}", self.alpha_const)
        print_and_log("weight_sdev          {}", self.weight_sdev)
        print_and_log("==================================================================================")
        for layer in range(0, self.layer_ordinal + 1):
            print_and_log("{}       {}          {}",layer ,self.NNlayer[layer][1], self.NNlayer[layer][0].shape)
        print_and_log("==================================================================================")
        print_and_log("Network Size:    {}", network_size)        
        print_and_log("==================================================================================")
        
    def create_network(self, x, y_true, dropout):
        # CREATE THE LAYERS
        self.NNlayer.append([x, "input            "])
        for layer in range(0,self.conv_layers):
            next = convolution_layer(inputT=self.NNlayer[self.layer_ordinal][0], patchS=self.conv_patch_size, layerS=self.conv_layer_size[layer], actFunc=self.conv_activation)
            self.NNlayer.append([next, "convolution_layer"])
            self.layer_ordinal += 1
            next = convolution_layer(inputT=self.NNlayer[self.layer_ordinal][0], patchS=self.conv_patch_size, layerS=self.conv_layer_size[layer], actFunc=self.conv_activation)
            self.NNlayer.append([next, "convolution_layer"])
            self.layer_ordinal += 1
            next = max_pool_2x2(inputT=self.NNlayer[self.layer_ordinal][0], strideS=self.max_pool_stride, frac_ratio=self.max_pool_frac_ratio)
            self.NNlayer.append([next, "max_pool_2x2     "])
            self.layer_ordinal += 1
            next = tf.nn.dropout(self.NNlayer[self.layer_ordinal][0], keep_prob=dropout)
            self.NNlayer.append([next, "dropout          "])
            self.layer_ordinal += 1

        # FLATEN CNN OUTPUT TO FIT DENSE
        next = flaten_4Dto2D(self.NNlayer[self.layer_ordinal][0])
        self.NNlayer.append([next, "flaten_4Dto2D    "])
        self.layer_ordinal += 1

        # DENSE Layers (INCL. DROPOUT)
        for layer in range(0,self.dense_layers):
            next = dense_layer(self.NNlayer[self.layer_ordinal][0], layerS=self.dense_layers_size[layer], actFunc=self.dense_activation)
            self.NNlayer.append([next, "dense_layer      "])
            self.layer_ordinal += 1
            next = tf.nn.dropout(self.NNlayer[self.layer_ordinal][0], keep_prob=dropout)
            self.NNlayer.append([next, "dropout          "])
            self.layer_ordinal += 1

        self.print_model_params(x, y_true, dropout)
        # Y PREDICTIONS (ULTIMATE OUTPUT)
        return self.NNlayer[self.layer_ordinal][0]


