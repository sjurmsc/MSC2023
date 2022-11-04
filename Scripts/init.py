"""
Initializes the task to be done on Odin

this script will contain all instructions for which networks to run on odin
--runs all code that has to do with different data permutations

All git operations will happen in this script
"""

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

# External packages
import json
import time
from pathlib import Path
import segyio
from PIL import Image
import numpy as np
from keras.models import load_model
import tensorflow as tf
from sklearn.model_selection import train_test_split

# My scripts
from Log import *
from Architectures import *
from Feat_aug import *

class RunModels:
    """
    Takes settings and runs a model based on it
    """
    def __init__(self, train_data, test_data, config, config_range=None):
        self.config = config
        self.config_range = config_range
        self.model_name_gen = give_modelname()
        self.train_data = train_data
        self.test_data = test_data

        
        if len(self.train_data) == 2:
            target, traces = self.train_data
            self.target_st_dev = stats.tstd(target, axis=None)
            target_norm = mpl.colors.Normalize(-self.target_st_dev, self.target_st_dev)
            self.target_cmap = lambda x : plt.cm.plasma(target_norm(x))
        elif len(self.train_data) == 1:
            traces = self.train_data
        self.seis_st_dev = stats.tstd(traces, axis=None)
        seis_norm = mpl.colors.Normalize(-self.seis_st_dev, self.seis_st_dev)
        self.seis_cmap = lambda x : plt.cm.seismic(seis_norm(x))
        


    def modelname(self):
        pass

    def objective(self, trial):
        groupname, modelname = next(self.model_name_gen)
        tbdir = './_tb/{}/{}'.format(groupname, modelname)
        os.makedirs(tbdir, exist_ok=True)
        self.tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tbdir, histogram_freq=1)

        sfunc = dict()
        sfunc['float'], sfunc['int'], sfunc['categorical'] = [trial.suggest_float, trial.suggest_int, trial.suggest_categorical]

        for key, items in config_range.items():
            suggest_func = sfunc[items[0]]
            self.config[key] = suggest_func(key, *items[1])

        model, History = compiled_TCN(self.train_data, self.config, callbacks=self.tb_callback)

        # Saving the model
        model_loc = './Models/{}/{}'.format(groupname, modelname)

        if not os.path.isdir(model_loc):
            os.mkdir(model_loc)
        model.save(model_loc)

        # Evaluating the model
        test_data = [self.test_data[0], self.test_data[1][:len(self.test_data[0])]]
        X, Y = test_data[1], test_data
        error = model.evaluate(X, Y, batch_size = 1, verbose=0)
        tot_error, reg_error, rec_error = error
        # Image
        seis_cmap = self.seis_cmap
        ai_cmap = self.target_cmap
        
        p, pt = create_pred_image_from_1d(model, self.train_data)

        if not os.path.isdir('./TEMP'):
            os.mkdir('./TEMP')
        image_folder = './TEMP/{}'.format(groupname)
        if not os.path.isdir(image_folder):
            os.makedirs(image_folder, exist_ok=True)
        
        # Image with comparisons
        p_name = image_folder + '/{}_combined_pred.jpg'.format(modelname)
        im_p = ai_cmap(p)
        img_p = Image.fromarray((im_p[:, :, :3]*255).astype(np.uint8)).save(p_name)

        if update_scores('{}/{}'.format(groupname, modelname), rec_error):
            replace_md_image(p_name, rec_error)

        with open(model_loc + '/' + 'config.json', 'w') as w_file:
            w_file.write(json.dumps(config, indent=2))
        save_training_progression(History.history, model_loc)

        return reg_error + rec_error


import matplotlib.pyplot as plt
import matplotlib as mpl
from glob import glob
import os.path
import numpy as np
import scipy.stats as stats
import statistics
from itertools import product, permutations
import optuna


# Functions
class config_iterator:
    def __init__(self, config, variable_config):
        self.config = config
        self.variable_config = variable_config
        self.keys = variable_config.keys()

        self.it = product(*variable_config.values())

    def __next__(self):
        try:
            vals = next(self.it)
        except:
            return None
        for key, val in zip(self.keys, vals):
            config[key] = val
        return config




if __name__ == '__main__':
    use_optuna = True ; n_trials = 50
    # Load data

    OD_fp = Path('../OneDrive - NGI/Documents/NTNU/MSC_DATA')
    seis_data_fp = r'..\OneDrive - NGI\Documents\NTNU\MSC_DATA\TNW_B02_5110_MIG_DPT.sgy' # Location to seismic data
    ai_data = r'..\OneDrive - NGI\Documents\NTNU\MSC_DATA\TNW_B02_5110_MIG.Abs_Zp.sgy'

    traces, z = get_traces(seis_data_fp)
    traces = traces[:, :]
    ai, z = get_traces(ai_data)
    ai = ai[:, :]

    # Splitting into test and training data for naive comparison
    split_loc = traces.shape[0]//2
    TRAINDATA = traces[:split_loc]
    TESTDATA = traces[split_loc:]
    
    ai_TRAINDATA = ai[:split_loc]
    ai_TESTDATA = ai[split_loc:]
    
    train_data =    [ai_TRAINDATA, TRAINDATA.reshape((len(TRAINDATA), len(TRAINDATA[0]), 1))]
    test_data =     [ai_TESTDATA, TESTDATA.reshape((len(TESTDATA), len(TESTDATA[0]), 1))]

    # Must structure the data into an array format
    ol = 2
    width_shape = 10
    height_shape = 500
    upper_bound = 600

    # train_data = split_image_into_data_packets(TRAINDATA, (width_shape, height_shape), upper_bound=upper_bound, overlap=ol)
    # test_data = split_image_into_data_packets(TESTDATA, (width_shape, height_shape), upper_bound=upper_bound, overlap=ol)
    #print(train_data.shape)

    # Visuals
    """
    Needs normalization within the standard deviations of the population
    """
    seis_st_dev = stats.tstd(traces, axis=None)
    norm = mpl.colors.Normalize(-seis_st_dev, seis_st_dev)
    cmap = lambda x : plt.cm.seismic(norm(x))


    # CONFIG
    config = dict()
    config['nb_filters']            = 2
    config['kernel_size']           = 8 # JR used 5
    config['dilations']             = [1, 2, 4, 8, 16, 32]
    config['padding']               = 'same'
    config['use_skip_connections']  = True
    config['dropout_rate']          = 0.01
    config['return_sequences']      = True
    config['activation']            = 'relu'
    config['convolution_type']      = 'Conv1D'
    config['learn_rate']            = 0.03
    config['kernel_initializer']    = 'he_normal'
    config['use_batch_norm']        = False
    config['use_layer_norm']        = False
    config['use_weight_norm']       = True

    config['batch_size']            = 20
    config['epochs']                = 12
    config['convolution_depth']     = 2

    # Iteratives
    makemodel = True; loadmodel = not makemodel

    if makemodel:
        model_name_gen = give_modelname()

        if use_optuna:
            # First using a couple of demonstrative values
            config_range = dict()
            # Floats
            config_range['learn_rate']      = ('float', (0.005, 0.05))
            config_range['dropout_rate']    = ('float', (0.01, 0.1))

            # Ints
            config_range['nb_filters']      = ('int', (1, 8))
            config_range['batch_size']      = ('int', (20, 40))
            config_range['epochs']          = ('int', (10, 100))

            # Categoricals
            #config_range['padding']         = ('categorical', (['causal', 'same'],))


            R = RunModels(train_data, test_data, config, config_range)
            study = optuna.create_study()
            study.optimize(R.objective, n_trials=n_trials)

        else:
            variable_config = dict()
            variable_config['nb_filters'] = [1, 2]
            variable_config['kernel_size'] = [8]
            config_iter = config_iterator(config, variable_config)

            config = next(config_iter)

            while config != None:
                groupname, modelname = next(model_name_gen)
                model, History = compiled_TCN(train_data, config)
                
                model_loc = './Models/{}/{}'.format(groupname, modelname)
                if not os.path.isdir(model_loc):
                    os.mkdir(model_loc)
                model.save(model_loc)
                with open(model_loc + '/' + 'config.json', 'w') as w_file:
                    w_file.write(json.dumps(config))
                save_training_progression(History.history, model_loc)
                config = next(config_iter)
            
    if loadmodel:
        groupname = 'ABA'
        model = load_model('./Models/{}/0'.format(groupname))
        

        test_data = [test_data[0], test_data[1][:len(test_data[0])]]
        X, Y = test_data[1], test_data

        error = model.evaluate(X, Y, batch_size = 1, verbose=0)
        print(error)
    
    
    # histogram_data = (pt[0].flatten(), pt[1].flatten())
    
    # colors = ['orange', 'b']

    # plt.hist(histogram_data, 20, density=True, color=colors)
    # plt.show()

    # scores = model.evaluate(test_data, test_data)
    # print('score: {}'.format(scores))

