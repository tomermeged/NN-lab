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
CONV_LAYER_SIZE = [96, 96, 96, 96, 64, 64]
CONV_PATCH_SIZE = 4
CONV_ACTIVATION = LEAKY_RELU

MAX_POOL_STRIDE = 0
MAX_POOL_FRAC_RATIO = [ON, 1.25, 1.25]

DENSE_LAYERS = 1
DENSE_LAYERS_SIZE = [800, 800]
DENSE_ACTIVATION = LEAKY_RELU

BIAS_CONST = 0.1
ALPHA_CONST = 0.2
WEIGHT_SDEV = 0.1
WEIGHT_MEAN = 0.0001
INIT_VARS = [BIAS_CONST, ALPHA_CONST, WEIGHT_SDEV, WEIGHT_MEAN]


#################################################
      
class CnnLecun:

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
        self.init_vars = INIT_VARS
        

    def set_model_params(   self,
                            conv_layers = CONV_LAYERS,
                            conv_layer_size = CONV_LAYER_SIZE,
                            conv_patch_size = CONV_PATCH_SIZE,
                            conv_activation = CONV_ACTIVATION,
                            max_pool_stride = MAX_POOL_STRIDE,
                            max_pool_frac_ratio = MAX_POOL_FRAC_RATIO,
                            dense_layers = DENSE_LAYERS,
                            dense_layers_size =  DENSE_LAYERS_SIZE,
                            dense_activation = DENSE_ACTIVATION,
                            init_vars = INIT_VARS):
                            
        self.conv_layers = conv_layers
        self.conv_layer_size = conv_layer_size
        self.conv_patch_size = conv_patch_size
        self.conv_activation = conv_activation
        self.max_pool_stride = max_pool_stride
        self.max_pool_frac_ratio = max_pool_frac_ratio
        self.dense_layers = dense_layers
        self.dense_layers_size =  dense_layers_size
        self.dense_activation = dense_activation
        self.init_vars = init_vars
    
    
    def print_model_params(self):
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
        print_and_log("bias_const               {}", self.init_vars[0])
        print_and_log("alpha_const              {}", self.init_vars[1])
        print_and_log("weight_sdev              {}", self.init_vars[2])    
        print_and_log("==================================================================================")
        for layer in range(0, self.layer_ordinal+1):
            print_and_log("{}       {}          {}",layer ,self.NNlayer[layer][1], self.NNlayer[layer][0].shape)
        print_and_log("==================================================================================")
        print_and_log("Network Size:    {}", network_size)        
        print_and_log("==================================================================================")
        
    def create_network(self, x, y_true, dropout):
        # CREATE THE LAYERS
        self.NNlayer.append([x, "input            "])
        
        for layer in range(0,self.conv_layers):
            next = convolution_layer(inputT=self.NNlayer[self.layer_ordinal][0], patchS=self.conv_patch_size, layerS=self.conv_layer_size[layer], actFunc=self.conv_activation, varInit=self.init_vars)
            self.NNlayer.append([next, "convolution_layer"])
            self.layer_ordinal += 1
            next = max_pool_2x2(inputT=self.NNlayer[self.layer_ordinal][0], strideS=self.max_pool_stride, frac_ratio=self.max_pool_frac_ratio)
            self.NNlayer.append([next, "max_pool_2x2     "])
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

        self.print_model_params()
        # Y PREDICTIONS (ULTIMATE OUTPUT)
        return self.NNlayer[self.layer_ordinal][0]


