"""
Initializes the task to be done on Odin

this script will contain all instructions for which networks to run on odin
--runs all code that has to do with different data permutations

All git operations will happen in this script
"""
# External packages
import json
import time
from pathlib import Path
import segyio

# My scripts
from Log import *
from Architectures import *

class RunModel:
    """
    Takes settings and runs a model based on it
    """
    def __init__(settings):
        pass


settings = {}
"""
In settings must be:
network: f.ex 2DTCN, 1DTCN, 2DTCN_WS [weight sharing], Randomforest ... etc
# What the network is, will affect what the init script uses to initialize the model

if TCN networks:
    dropout:
    kernel_size
    filters
    loss
    dilation
    

for all networks:
    dataset: contains the label names that can be used as keys for retrieving file paths
            for getting the data [file paths must be robust, and located either on p: or in the
            repo]
    epochs: 
    batches:
    learn_rate:

for training where certain fields are not needed, these may be filled with None, or be omitted
"""


control = {}
control['settings'] = settings.copy() # settings
control['summary_stats'] = {} # to be filled in later
from matplotlib.pyplot import imshow, show

if __name__ == '__main__':
    # Load data
    seis_data_fp = r'C:\Users\SjB\OneDrive - NGI\Documents\NTNU\MSC_DATA\TNW_B02_5110_MIG_DPT.sgy' # Location to seismic data
    with segyio.open(seis_data_fp) as seis_data:
        traces = segyio.collect(seis_data.trace)

    split_loc = len(traces)//2
    TRAINDDATA = traces[:split_loc]
    TESTDATA = traces[split_loc:]


    # Må dele inn datasettet i trening og testing

    # CONFIG
    config = dict()
    config['nb_filters']            = 64
    config['kernel_size']           = 8 # JR used 5
    config['dilations']             = [1, 2, 4, 8, 16]
    config['padding']               = 'causal'
    config['use_skip_connections']  = True
    config['dropout_rate']          = 0.1
    config['return_sequences']      = True
    config['activation']            = 'relu'
    config['convolution_type']      = 'Conv2D'
    config['learn_rate']            = 0.02
    config['kernel_initializer']    = 'he_normal'
    config['use_batch_norm']        = False
    config['use_layer_norm']        = False
    config['use_weight_norm']       = True

    # ML
    model = compiled_TCN(train_data, config)
    scores = model.evaluate(test_data_X, test_data_Y)
    print('score: {}'.format(scores))
    
