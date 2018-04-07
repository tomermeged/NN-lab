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

import CNN_lecun

#################################################
# MODEL SPECIFIC PARAMS

NUM_NETWORKS = 3
DENSE_LAYERS_SIZE = 1024
DENSE_ACTIVATION = LEAKY_RELU

#################################################
  
class CnnLecunBoost:

    def __init__(self):
        self.Cnn = []
        self.y_pred = []
        self.NNlayer = []
        self.layer_ordinal = 0
        self.num_networks = NUM_NETWORKS
        self.dense_layers_size = DENSE_LAYERS_SIZE
        self.dense_activation = DENSE_ACTIVATION

    def print_model_params(self):
        network_size = calc_params()
        print_and_log("==================================================================================")
        print_and_log("num_networks          {}", self.num_networks)
        print_and_log("==================================================================================")
        for layer in range(0, self.layer_ordinal+1):
            print_and_log("{}       {}          {}",layer ,self.NNlayer[layer][1], self.NNlayer[layer][0].shape)
        print_and_log("==================================================================================")
        print_and_log("Network Size:    {}", network_size)        
        print_and_log("==================================================================================")
        
    def create_network(self, x, y_true, dropout):
        for i in range (self.num_networks):
            self.Cnn.append(CNN_lecun.CnnLecun())
            self.Cnn[i].set_model_params(   bias_const = 0.1*(i+1), 
                                            alpha_const = 0.2*(i+1),
                                            weight_sdev = 0.1*(i+1))
            self.y_pred.append(self.Cnn[i].create_network(x, y_true, dropout))
        
        y_pred_concat = tf.concat(self.y_pred[:], -1)
        self.NNlayer.append([y_pred_concat, "concat_layer            "])
        next = dense_layer(self.NNlayer[self.layer_ordinal][0], layerS=self.dense_layers_size, actFunc=self.dense_activation)
        self.NNlayer.append([next, "dense_layer      "])
        self.layer_ordinal += 1
        next = tf.nn.dropout(self.NNlayer[self.layer_ordinal][0], keep_prob=dropout)
        self.NNlayer.append([next, "dropout          "])
        self.layer_ordinal += 1
        self.print_model_params()
        return self.NNlayer[self.layer_ordinal][0]
   
    
    
    
    
