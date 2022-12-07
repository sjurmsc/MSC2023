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
from keras.utils.vis_utils import plot_model
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
from glob import glob
import os.path
import numpy as np
import scipy.stats as stats
from itertools import product
import optuna

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
        self.seis_testimage_fp = "../OneDrive - NGI/Documents/NTNU/MSC_DATA/2DUHRS_06_MIG_DEPTH/TNW_B02_5110_MIG_DPT.sgy"
        self.ai_testimage_fp = "../OneDrive - NGI/Documents/NTNU/MSC_DATA/00_AI/TNW_B02_5110_MIG.Abs_Zp.sgy"

        if len(self.train_data) == 2:
            traces, train_y = self.train_data
            if len(train_y) == 2:
                flat_target = train_y[0].flatten()
            elif len(train_y) == 1:
                flat_target = train_y.flatten()
            else: raise ValueError('Unexpected dimansionality of target')

            self.target_max = np.max(flat_target, axis=None)
            self.target_min = np.min(flat_target[np.nonzero(flat_target)])

            target_norm = mpl.colors.Normalize(self.target_min, self.target_max)
            self.target_cmap = lambda x : plt.cm.plasma(target_norm(x))
        elif len(self.train_data) == 1:
            traces = self.train_data
            raise ValueError('This part should not run')

        self.seis_st_dev = stats.tstd(traces, axis=None)
        seis_norm = mpl.colors.Normalize(-self.seis_st_dev, self.seis_st_dev)
        self.seis_cmap = lambda x : plt.cm.seismic(seis_norm(x))

    def objective(self, trial):
        groupname, modelname = next(self.model_name_gen)
        
        tbdir = './_tb'
        self.tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tbdir, histogram_freq=1)

        sfunc = dict()
        sfunc['float'], sfunc['int'], sfunc['categorical'] = [trial.suggest_float, trial.suggest_int, trial.suggest_categorical]

        for key, items in config_range.items():
            suggest_func = sfunc[items[0]]
            self.config[key] = suggest_func(key, *items[1])


        model, History = compiled_TCN(self.train_data, self.config, callbacks=[self.tb_callback])

        # Saving the model
        model_loc = './Models/{}/{}'.format(groupname, modelname)

        if not os.path.isdir(model_loc): os.mkdir(model_loc)
        model.save(model_loc)
        plot_model(model, to_file=model_loc+'/model.png', show_shapes=True, show_layer_names=True)

        # Evaluating the model
        X, Y = test_data
        error = model.evaluate(X, Y, batch_size = 1, verbose=0)
        tot_error, reg_error, rec_error = error
        
        # Image
        seis_cmap = self.seis_cmap
        ai_cmap = self.target_cmap
        
        seis_testimage, ai_testimage, _ = get_matching_traces(self.seis_testimage_fp, self.ai_testimage_fp, group_traces=self.config['group_traces'], trunc=80)
        target_pred, recon_pred = create_pred_image(model,  [seis_testimage, ai_testimage])
        create_ai_error_image(target_pred-ai_testimage, seis_testimage, filename=model_loc+'/error_image.png')
        #prediction_histogram(pt[0], pt[1], bins=500)

        if not os.path.isdir('./TEMP'): os.mkdir('./TEMP')
        image_folder = './TEMP/{}'.format(groupname)
        if not os.path.isdir(image_folder): os.makedirs(image_folder, exist_ok=True)
        
        # Image with comparisons
        p_name = image_folder + '/{}_combined_target_pred.jpg'.format(modelname)
        rec_p_name = image_folder + '/{}_combined_recon_pred.jpg'.format(modelname)
        im_p = ai_cmap(target_pred)
        im_rec_p = seis_cmap(recon_pred)
        Image.fromarray((im_p[:, :, :3]*255).astype(np.uint8)).save(p_name)
        Image.fromarray((im_rec_p[:, :, :3]*255).astype(np.uint8)).save(rec_p_name)

        if update_scores('{}/{}'.format(groupname, modelname), rec_error):
            replace_md_image(p_name, rec_error)

        with open(model_loc + '/' + 'config.json', 'w') as w_file:
            w_file.write(json.dumps(config, indent=2))
        save_training_progression(History.history, model_loc)

        del model # Clear up the memory location for next model
        return reg_error + rec_error


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
    use_optuna = True ; n_trials = 40
    makemodel = True; loadmodel = not makemodel

    # CONFIG
    config = dict()
    config['nb_filters']            = 8
    config['kernel_size']           = (3, 8) # Height, width
    config['dilations']             = [1, 2, 4, 8, 16, 32]
    config['padding']               = 'same'
    config['use_skip_connections']  = True
    config['dropout_type']          = 'normal'
    config['dropout_rate']          = 0.03
    config['return_sequences']      = True
    config['activation']            = 'relu'
    config['convolution_type']      = 'Conv1D'
    config['learn_rate']            = 0.001
    config['kernel_initializer']    = 'he_normal'

    config['use_batch_norm']        = False
    config['use_layer_norm']        = False
    config['use_weight_norm']       = True

    config['nb_tcn_stacks']         = 3
    config['nb_reg_stacks']         = 5
    config['nb_rec_stacks']         = 3    

    config['batch_size']            = 20
    config['epochs']                = 1

    config['seismic_data']          = ['2DUHRS_06_MIG_DEPTH']
    config['ai_data']               = ['00_AI']
    config['cpt_data']              = ['']
    config['group_traces']          = 1

    if config['group_traces']>1: config['convolution_type'] = 'Conv2D'
    else: config['kernel_size'] = config['kernel_size'][1]

    # Retrieving the data
    seismic_datasets =  list(config['seismic_data'])
    ai_datasets =       list(config['ai_data'])
    cpt_datasets =      list(config['cpt_data'])
    group_traces =      config['group_traces']

    if len(ai_datasets):
        train_data, test_data, scalers = sgy_to_keras_dataset(seismic_datasets, 
                                                              ai_datasets, 
                                                              fraction_data=0.06, 
                                                              test_size=0.8, 
                                                              group_traces=group_traces, 
                                                              X_normalize='StandardScaler',
                                                              y_normalize='MinMaxScaler',
                                                              truncate_data=100)
        test_X, test_y = test_data

    # elif len(cpt_datasets):
    #     raise ValueError('This should not be populated yet')

    if makemodel:
        model_name_gen = give_modelname()  # Initiate iterator to give model names

        if use_optuna:
            # First using a couple of demonstrative values
            config_range = dict()
            
            # Floats
            config_range['learn_rate']      = ('float', (0.005, 0.05))
            config_range['dropout_rate']    = ('float', (0.01, 0.1))

            # Ints
            # config_range['nb_filters']      = ('int', (2, 12))
            # config_range['nb_tcn_stacks']   = ('int', (1, 3))
            # config_range['kernel_size']     = ('int', (6, 12))
            config_range['batch_size']      = ('int', (20, 30))
            config_range['epochs']          = ('int', (90, 150))

            # Categoricals
            #config_range['padding']         = ('categorical', (['causal', 'same'],))


            R = RunModels(train_data, test_data, config, config_range)
            study = optuna.create_study()
            study.optimize(R.objective, n_trials=n_trials)


        else:
            variable_config = dict()
            variable_config['nb_filters'] = [2]
            variable_config['kernel_size'] = [8]
            config_iter = config_iterator(config, variable_config)

            config = next(config_iter)

            while config != None:
                groupname, modelname = next(model_name_gen)

                tbdir = './_tb'
                tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tbdir, histogram_freq=1)

                model, History = compiled_TCN(train_data, config, callbacks = [tb_callback])
                
                model_loc = './Models/{}/{}'.format(groupname, modelname)
                if not os.path.isdir(model_loc):
                    os.mkdir(model_loc)
                model.save(model_loc)
                with open(model_loc + '/' + 'config.json', 'w') as w_file:
                    w_file.write(json.dumps(config))

                save_training_progression(History.history, model_loc)

                config = next(config_iter)

    if loadmodel:
        mname = 'ABA/0'
        model = load_model('./Models/{}'.format(mname))
        error = model.evaluate(test_X, test_y, batch_size = 1, verbose=0)
        print(error)
