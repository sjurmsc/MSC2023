"""
This script is to be run from the command line, and initializes the instructions
to be performed for any training instance (denoted by three upper case letters).

The configurations to the machine learning, and the procedures for iterating
over these configurations are also defined in this script. 
"""

# External packages
from pathlib import Path
from PIL import Image
import numpy as np
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import os.path
from shutil import rmtree
import numpy as np
import scipy.stats as stats
import optuna

# My scripts
from Log import *
from Architectures import *
from Feat_aug import *

class RunModels:
    """
    This class is used by optuna to permute config parameters in the models in an optimal way
    """
    def __init__(self, train_data, test_data, config, config_range=None, scalers=False):
        self.config = config
        self.config_range = config_range
        self.model_name_gen = give_modelname()
        self.train_data = train_data
        self.test_data = test_data
        self.seis_testimage_fp = "../OneDrive - NGI/Documents/NTNU/MSC_DATA/2DUHRS_06_MIG_DEPTH/TNW_B02_5110_MIG_DPT.sgy"
        self.ai_testimage_fp = "../OneDrive - NGI/Documents/NTNU/MSC_DATA/00_AI/TNW_B02_5110_MIG.Abs_Zp.sgy"
        
        # Set up directory to log tensorboard
        self.tbdir = Path('./_tb')
        if self.tbdir.exists(): rmtree(str(self.tbdir), ignore_errors=True); self.tbdir.mkdir()
        
        # Set up image folder
        if not os.path.isdir('./TEMP'): os.mkdir('./TEMP')

        # Load scalers
        if scalers:
            self.X_scaler, self.y_scaler = scalers

        # Create colormaps for plotting
        if len(self.train_data) == 2: # If the task is not just reconstruction
            traces, train_y = self.train_data

            # Making the code resilient to different dimensionality of predictions; may be only target, only reconstruct or both
            # if len(train_y) == 2: 
            #     flat_target = train_y[0].flatten()
            # elif len(train_y) == 1:
            #     flat_target = train_y.flatten()
            # else: raise ValueError('Unexpected dimansionality of target')     %%%%%%%%%% Code may not be needed because inputs are scaled

            self.target_cmap = lambda x : plt.cm.plasma(x) # Target parameter colormap
        elif len(self.train_data) == 1:
            traces = self.train_data
            raise ValueError('This part should not run')

        # Getting colormap for plotting
        self.seis_st_dev = stats.tstd(traces, axis=None)
        seis_norm = mpl.colors.Normalize(-self.seis_st_dev, self.seis_st_dev) # Seismic plot is scaled to std dev in the data
        self.seis_cmap = lambda x : plt.cm.seismic(seis_norm(x))

    def update_config_with_suggestions(self, trial):
        if not isinstance(self.sfunc, dict):
            self.sfunc = dict()
            self.sfunc['float'], self.sfunc['int'], self.sfunc['categorical'] = [trial.suggest_float, trial.suggest_int, trial.suggest_categorical]

        for key, items in self.config_range.items():
            suggest_func = self.sfunc[items[0]]
            self.config[key] = suggest_func(key, *items[1])

    def objective(self, trial):
        # Update name of the model and configs for model run
        groupname, modelname = next(self.model_name_gen)
        self.update_config_with_suggestions(trial)

        # New callback for this model instance
        self.tb_callback = tf.keras.callbacks.TensorBoard(log_dir=str(self.tbdir), histogram_freq=1)

        # Call the model to fit on training data
        model, History = compiled_TCN(self.train_data, self.config, callbacks=[self.tb_callback])

        # Saving the model
        model_loc = './Models/{}/{}'.format(groupname, modelname)
        if not os.path.isdir(model_loc): os.mkdir(model_loc)
        #model.save(model_loc)

        # Plotting the model for later analysis
        #plot_model(model.generator, to_file=model_loc+'/model.png', show_shapes=True, show_layer_names=True, expand_nested=True, show_layer_activations=True)

        # Evaluating the model
        X, Y = test_data
        error = model.evaluate(X, Y, batch_size = 20, verbose=2, steps=40)
        
        # First metric should always be loss
        if isinstance(error, list):
            tot_error = error[0]
        
        # Image colormaps
        seis_cmap = self.seis_cmap
        ai_cmap = self.target_cmap
        
        # Have to get the traces here, because groupings may change
        seis_testimage, ai_testimage, _ = get_matching_traces(self.seis_testimage_fp,
                                                              self.ai_testimage_fp, 
                                                              group_traces=self.config['group_traces'], 
                                                              trunc=80)
        
        # Scale the test images with dataset scalers
        old_X_shp = seis_testimage.shape; old_y_shp = ai_testimage.shape
        seis_testimage = self.X_scaler.transform(seis_testimage.reshape(-1, 1)); ai_testimage = self.y_scaler.transform(ai_testimage.reshape(-1, 1))
        seis_testimage = seis_testimage.reshape(old_X_shp); ai_testimage = ai_testimage.reshape(old_y_shp)

        # With the testimage create a prediction images for visual inspection
        target_pred, recon_pred, target_pred_diff = create_pred_image(model,  [seis_testimage, ai_testimage])
        #create_ai_error_image((target_pred_diff)**2, seis_testimage, filename=model_loc+'/error_image.png')

        image_folder = './TEMP/{}'.format(groupname)
        if not os.path.isdir(image_folder): os.makedirs(image_folder, exist_ok=True)
        
        # Image with comparisons
        p_name = image_folder + '/{}_combined_target_pred.jpg'.format(modelname)
        rec_p_name = image_folder + '/{}_combined_recon_pred.jpg'.format(modelname)

        # Applying colormaps to the images
        im_p = ai_cmap(target_pred); im_rec_p = seis_cmap(recon_pred)

        # Saving the images
        Image.fromarray((im_p[:, :, :3]*255).astype(np.uint8)).save(p_name)
        Image.fromarray((im_rec_p[:, :, :3]*255).astype(np.uint8)).save(rec_p_name)

        # Update scoreboard for repo image
        # if update_scores('{}/{}'.format(groupname, modelname), rec_error):
        #     replace_md_image(p_name, rec_error)

        # Save data about model params and training history
        save_config(model_loc, config)
        save_training_progression(History.history, model_loc)

        del model # Clear up the memory location for next model
        return tot_error


from itertools import product

class config_iterator:
    """
    Custom class made to iterate through the different permutations
    of model configurations
    """
    def __init__(self, config, variable_config):
        self.config = config
        self.variable_config = variable_config
        self.keys = variable_config.keys()

        # Setting up the permutation iterator
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
    use_optuna = False ; n_trials = 1
    makemodel = True

    # CONFIG :: Static configurations that get replaced by variations
    config = dict()
    config['nb_filters']            = 12
    config['kernel_size']           = (3, 9) # Height, width
    config['dilations']             = [1, 2, 4, 8, 16, 32]
    config['padding']               = 'same'
    config['use_skip_connections']  = True
    config['dropout_type']          = 'normal'
    config['dropout_rate']          = 0.03
    config['return_sequences']      = True
    config['activation']            = LeakyReLU()
    config['convolution_func']      = Conv1D
    config['learn_rate']            = 0.001
    config['kernel_initializer']    = 'he_normal'

    config['use_batch_norm']        = False
    config['use_layer_norm']        = False
    config['use_weight_norm']       = True
    config['use_adversaries']       = True
    if config['use_adversaries']:
        config['alpha']     = 0.6
        config['beta']      = 0.4

    config['nb_tcn_stacks']         = 3
    config['nb_reg_stacks']         = 5
    config['nb_rec_stacks']         = 3    

    config['batch_size']            = 20
    config['epochs']                = 1

    config['seismic_data']          = ['2DUHRS_06_MIG_DEPTH']
    config['ai_data']               = ['00_AI']
    config['cpt_data']              = ['']
    config['group_traces']          = 1

    if config['group_traces']>1: config['convolution_func'] = Conv2D
    else: config['kernel_size'] = config['kernel_size'][1]

    # Optuna config values
    config_range = dict()
            
    ######### Floats #########
    config_range['learn_rate']      = ('float', (0.005, 0.05))
    config_range['dropout_rate']    = ('float', (0.01, 0.1))

    ########## Ints ##########
    # config_range['nb_filters']      = ('int', (2, 12))
    # config_range['nb_tcn_stacks']   = ('int', (1, 3))
    # config_range['kernel_size']     = ('int', (6, 12))
    config_range['batch_size']      = ('int', (20, 30))
    config_range['epochs']          = ('int', (40, 60))

    ###### Categoricals ######
    #config_range['padding']         = ('categorical', (['causal', 'same'],))


    # "Manual" permutation of the config values
    variable_config = dict()
    variable_config['nb_filters'] = [2]
    variable_config['kernel_size'] = [8]


    # Retrieving the data
    seismic_datasets =  list(config['seismic_data'])
    ai_datasets =       list(config['ai_data'])
    cpt_datasets =      list(config['cpt_data'])
    group_traces =      config['group_traces']

    if len(ai_datasets):
        train_data, test_data, scalers = sgy_to_keras_dataset(seismic_datasets, 
                                                              ai_datasets, 
                                                              fraction_data=0.01, 
                                                              test_size=0.8, 
                                                              group_traces=group_traces, 
                                                              y_normalize='MinMaxScaler',
                                                              truncate_data=80)
        test_X, test_y = test_data


    # Creating the model
    if makemodel:
        model_name_gen = give_modelname()  # Initiate iterator to give model names

        # Optuna uses the RunModels class
        if use_optuna:
            R = RunModels(train_data, 
                          test_data, 
                          config, 
                          config_range=config_range, 
                          scalers=scalers)

            study = optuna.create_study()
            # Over time optuna minimizes the loss returned by the objective function
            study.optimize(R.objective, n_trials=n_trials)

        else: # Uses manual method
            config_iter = config_iterator(config, variable_config)
            config = next(config_iter)
            tbdir = './_tb'

            while config != None: # None is returne by iterator at the end of permutations
                groupname, modelname = next(model_name_gen)

                tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tbdir, histogram_freq=1)

                model, History = compiled_TCN(train_data, config, callbacks = [tb_callback])
                
                model_loc = './Models/{}/{}'.format(groupname, modelname)
                if not os.path.isdir(model_loc): os.mkdir(model_loc)
                # model.save(model_loc)

                save_config(model_loc, config)

                save_training_progression(History.history, model_loc)

                config = next(config_iter)

    # Loading the model of selected model name
    else: 
        mname = 'ABA/0'
        model = load_model('./Models/{}'.format(mname))
        error = model.evaluate(test_X, test_y, batch_size = 1, verbose=0)
        print(error)
