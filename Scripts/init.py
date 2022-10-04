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
import numpy as np
from keras.models import load_model

# My scripts
from Log import *
from Architectures import *
from Feat_aug import *

class RunModels:
    """
    Takes settings and runs a model based on it
    """
    def __init__(self, settings):
        self.group = new_group()
        self.init_time = datetime.now()
        pass

    def modelname(self):
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
from glob import glob
import os
import numpy as np

if __name__ == '__main__':
    # Load data
    seis_data_fp = r'C:\Users\SjB\OneDrive - NGI\Documents\NTNU\MSC_DATA\TNW_B02_5110_MIG_DPT.sgy' # Location to seismic data
    traces, z = get_traces(seis_data_fp)
    traces = traces[:, 600:1100]

    # Splitting into test and training data for naive comparison
    split_loc = traces.shape[0]//2
    TRAINDATA = traces[:split_loc]
    TESTDATA = traces[split_loc:]
    
    print(TRAINDATA.shape)
    train_data = TRAINDATA.reshape((len(TRAINDATA), len(TRAINDATA[0]), 1))
    test_data = TESTDATA.reshape((len(TESTDATA), len(TESTDATA[0]), 1))

    # Must structure the data into an array format
    ol = 100
    width_shape = 250
    height_shape = 500
    upper_bound = 600
    # train_data = split_image_into_data_packets(TRAINDATA, (width_shape, height_shape), upper_bound=upper_bound, overlap=ol)
    # test_data = split_image_into_data_packets(TESTDATA, (width_shape, height_shape), upper_bound=upper_bound, overlap=ol)

    # Exporting images to the TEMP folder %%%%%%%%%%%%%%% TEMPORARY
    do = False
    if do:
        cmap = plt.cm.get_cmap('seismic')
        image_folder = 'C:/Users/SjB/MSC2023/TEMP/seismic_images'
        files = glob(image_folder + '/*')
        for f in files:
            os.remove(f)

        for i, image in enumerate(train_data):
            f_name = image_folder + '/{}.jpg'.format(i)
            im = cmap(image.T)
            img = Image.fromarray((im[:, :, :3]*255).astype(np.uint8)).save(f_name)
        
            #norm = plt.Normalize(vmin = image.min(), vmax = image.max())
            #im = cmap(norm(image.T))
            #plt.imsave(image_folder+'\\{}.jpg'.format(i), im, cmap='seismic')
    # .%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    # test_data_X = test_data.copy()
    # test_data_Y = [val.flatten() for val in test_data_X]

    # CONFIG
    config = dict()
    config['nb_filters']            = 64
    config['kernel_size']           = 8 # JR used 5
    config['dilations']             = [1, 2, 4, 8, 16, 32]
    config['padding']               = 'causal'
    config['use_skip_connections']  = True
    config['dropout_rate']          = 0.15
    config['return_sequences']      = True
    config['activation']            = 'relu'
    config['convolution_type']      = 'Conv1D'
    config['learn_rate']            = 0.02
    config['kernel_initializer']    = 'he_normal'
    config['use_batch_norm']        = False
    config['use_layer_norm']        = False
    config['use_weight_norm']       = True

    
    # ML
    makemodel = False
    loadmodel = not makemodel

    if makemodel:
        model_name_gen = give_modelname()
        groupname, modelname = next(model_name_gen)
        model = compiled_TCN(train_data, config, epochs=12)
        
        model.save('./Models/{}/{}'.format(groupname, modelname))
    if loadmodel:
        grp = 'AAC'
        model = load_model('./Models/{}/0'.format(grp))
        p = create_pred_image_from_1d(model, train_data, train_data)
        cmap = plt.cm.get_cmap('seismic')
        image_folder = 'C:/Users/SjB/MSC2023/TEMP/{}'.format(grp)
        os.mkdir(image_folder)
        p_name = image_folder + '/pred.jpg'
        # t_name = image_folder + '/true.jpg'
        im_p = cmap(p)
        # im_t = cmap(t)
        img_p = Image.fromarray((im_p[:, :, :3]*255).astype(np.uint8)).save(p_name)
        # img_t = Image.fromarray((im_t[:, :, :3]*255).astype(np.uint8)).save(t_name)
        # replace_md_image('./{}'.format(p_name[21:]))


    scores = model.evaluate(train_data, train_data)
    print('score: {}'.format(scores))
    
