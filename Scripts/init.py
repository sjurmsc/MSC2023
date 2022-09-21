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
from PIL import Image

# My scripts
from Log import *
from Architectures import *
from Feat_aug import *

class RunModel:
    """
    Takes settings and runs a model based on it
    """
    def __init__(settings):
        pass

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
control['settings'] = {} # settings
control['summary_stats'] = {} # to be filled in later
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Load data
    seis_data_fp = r'C:\Users\SjB\OneDrive - NGI\Documents\NTNU\MSC_DATA\TNW_B02_5110_MIG_DPT.sgy' # Location to seismic data
    traces, _ = get_traces(seis_data_fp)

    # Splitting into test and training data for naive comparison
    split_loc = traces.shape[0]//2
    TRAINDATA = traces[:split_loc]
    TESTDATA = traces[split_loc:]
    

    # Must structure the data into an array format
    ol = 0
    width_shape = 100
    train_data = split_image_into_data_packets(TRAINDATA, (width_shape, 200), overlap=ol)
    test_data = split_image_into_data_packets(TESTDATA, (width_shape, 200), overlap=ol)

    cmap = plt.cm.seismic
    image_folder = r'C:\Users\SjB\MSC2023\TEMP\seismic_images'
    fig = plt.figure()
    for i, image in enumerate(train_data):
        fig.clf()
        norm = plt.Normalize(vmin = image.min(), vmax = image.max())
        im = cmap(norm(image.T))
        plt.imsave(image_folder+'\\{}.jpg'.format(i), im, cmap='seismic')

    test_data_X = test_data.copy()
    test_data_Y = [val.flatten() for val in test_data_X]
    # CONFIG
    config = dict()
    config['nb_filters']            = 64
    config['kernel_size']           = 8 # JR used 5
    config['dilations']             = [1, 2, 4, 8, 16]
    config['padding']               = 'same'
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
    
