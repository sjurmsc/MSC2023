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
    def __init__(self, train_data, test_data, config, config_range=None):
        self.config = config
        self.config_range = config_range
        self.model_name_gen = give_modelname()
        self.train_data = train_data
        self.test_data = test_data

        self.st_dev = stats.tstd(traces, axis=None)
        norm = mpl.colors.Normalize(-self.st_dev, self.st_dev)
        self.cmap = lambda x : plt.cm.seismic(norm(x))

    def modelname(self):
        pass

    def objective(self, trial):
        sfunc = dict()
        sfunc['float'], sfunc['int'], sfunc['categorical'] = [trial.suggest_float, trial.suggest_int, trial.suggest_categorical]

        for key, items in config_range.items():
            suggest_func = sfunc[items[0]]
            self.config[key] = suggest_func(key, *items[1])

        model = compiled_TCN(self.train_data, self.config)
        error = model.evaluate(self.test_data, self.test_data, verbose=0)
        groupname, modelname = next(self.model_name_gen)
        model_loc = './Models/{}/{}'.format(groupname, modelname)
        if not os.path.isdir(model_loc):
            os.mkdir(model_loc)
        model.save(model_loc)

        # Image
        cmap = self.cmap #plt.cm.get_cmap('seismic')  
        p, pt = create_pred_image_from_1d(model, self.train_data, self.train_data)

        image_folder = 'C:/Users/SjB/MSC2023/TEMP/{}'.format(groupname)
        if not os.path.isdir(image_folder):
            os.makedirs(image_folder, exist_ok=True)
        
        # Image with comparisons
        p_name = image_folder + '/{}_combined_pred.jpg'.format(modelname)
        im_p = cmap(p)
        img_p = Image.fromarray((im_p[:, :, :3]*255).astype(np.uint8)).save(p_name)

        replace_md_image(p_name)
        with open(model_loc + '/' + 'config.json', 'w') as w_file:
            w_file.write(json.dumps(config, indent=4))

        return error


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
    use_optuna = True
    # Load data
    seis_data_fp = r'C:\Users\SjB\OneDrive - NGI\Documents\NTNU\MSC_DATA\TNW_B02_5110_MIG_DPT.sgy' # Location to seismic data
    traces, z = get_traces(seis_data_fp)
    traces = traces[100:1100, 650:1000]

    # Splitting into test and training data for naive comparison
    split_loc = traces.shape[0]//2
    TRAINDATA = traces[:split_loc]
    TESTDATA = traces[split_loc:]
    
    
    train_data = TRAINDATA.reshape((len(TRAINDATA), len(TRAINDATA[0]), 1))
    test_data = TESTDATA.reshape((len(TESTDATA), len(TESTDATA[0]), 1))

    # Must structure the data into an array format
    ol = 2
    width_shape = 10
    height_shape = 500
    upper_bound = 600

    # train_data = split_image_into_data_packets(TRAINDATA, (width_shape, height_shape), upper_bound=upper_bound, overlap=ol)
    # test_data = split_image_into_data_packets(TESTDATA, (width_shape, height_shape), upper_bound=upper_bound, overlap=ol)
    print(train_data.shape)

    # Visuals
    """
    Needs normalization within the standard deviations of the population
    """
    st_dev = stats.tstd(traces, axis=None)
    norm = mpl.colors.Normalize(-st_dev, st_dev)
    cmap = lambda x : plt.cm.seismic(norm(x))


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
    global config
    config = dict()
    config['nb_filters']            = 2
    config['kernel_size']           = 8 # JR used 5
    config['dilations']             = [1, 2, 4, 8, 16, 32]
    config['padding']               = 'causal'
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

    # Iteratives
    
    
    # ML
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
            config_range['epochs']          = ('int', (10, 50))
            # Categoricals
            config_range['padding']         = ('categorical', (['causal', 'same'],))


            R = RunModels(train_data, test_data, config, config_range)
            study = optuna.create_study()
            study.optimize(R.objective, n_trials=50)

        else:
            variable_config = dict()
            variable_config['nb_filters'] = [1, 2]
            variable_config['kernel_size'] = [8]
            config_iter = config_iterator(config, variable_config)

            config = next(config_iter)

            while config != None:
                groupname, modelname = next(model_name_gen)
                model = compiled_TCN(train_data, config)
                
                model_loc = './Models/{}/{}'.format(groupname, modelname)
                if not os.path.isdir(model_loc):
                    os.mkdir(model_loc)
                model.save(model_loc)
                with open(model_loc + '/' + 'config.json', 'w') as w_file:
                    w_file.write(json.dumps(config))
                config = next(config_iter)
            
    if loadmodel:
        groupname = 'AAE'
        model = load_model('./Models/{}/0'.format(groupname))
        
        
    # cmap = plt.cm.seismic  
    
    # p, pt = create_pred_image_from_1d(model, train_data, train_data)

    
    
    # image_folder = 'C:/Users/SjB/MSC2023/TEMP/{}'.format(groupname)
    # if not os.path.isdir(image_folder):
    #     os.mkdir(image_folder)
    
    # # Image with comparisons
    # p_name = image_folder + '/combined_pred.jpg'
    # im_p = cmap(p)
    # img_p = Image.fromarray((im_p[:, :, :3]*255).astype(np.uint8)).save(p_name)
    # replace_md_image('./{}'.format(p_name[21:]))
    
    # histogram_data = (pt[0].flatten(), pt[1].flatten())
    
    # colors = ['y', 'b']

    # plt.hist(histogram_data, 20, density=True, color=colors)
    # plt.show()

    # scores = model.evaluate(test_data, test_data)
    # print('score: {}'.format(scores))

