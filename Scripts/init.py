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

    # CONFIG :: Static configurations that get replaced by variations
    config = dict()
    config['nb_filters']            = 12
    config['kernel_size']           = (3, 3) # Height, width
    config['padding']               = 'same'
    config['use_skip_connections']  = True
    config['dropout_type']          = 'normal'
    config['dropout_rate']          = 0.03
    config['return_sequences']      = True
    config['activation']            = LeakyReLU()
    config['learn_rate']            = 0.001
    config['kernel_initializer']    = 'he_normal'

    config['use_batch_norm']        = False
    config['use_layer_norm']        = False
    config['use_weight_norm']       = True


    config['batch_size']            = 20
    config['epochs']                = 1

    config['seismic_data']          = ['2DUHRS_06_MIG_DEPTH']
    config['ai_data']               = ['00_AI']
    config['cpt_data']              = ['']
    config['group_traces']          = 1


    # Retrieving the data
    seismic_datasets =  list(config['seismic_data'])
    ai_datasets =       list(config['ai_data'])
    cpt_datasets =      list(config['cpt_data'])

    # Creating the model



