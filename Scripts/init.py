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
from keras.utils.vis_utils import plot_model

from pandas import read_csv
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, cross_validate, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.manifold import TSNE
from sklearn.multioutput import MultiOutputRegressor

# My scripts
from Log import *
from Architectures import *
from Feat_aug import *
from NGI.GM_Toolbox import evaluate_modeldist_norm

# Configurations for the Random Forest model
RF_param_dict = {
    'max_depth': 20,
    'n_estimators': 20,
    'min_samples_leaf': 1,
    'min_samples_split': 4,
    'bootstrap': True,
    'criterion': 'mse'
    }

LGBM_param_dict = {
    'num_leaves': 31,
    'max_depth': 20,
    'n_estimators': 100
    }

NN_param_dict = {
    'epochs': 200,
    'batch_size': 20

    }

if __name__ == '__main__':

    # Retrieve data
    cpt_exhausive_dataset = read_csv(r'../OneDrive - NGI/Documents/NTNU/MSC_DATA/Database.csv')


    # Creating the model
    methods = ['Ensemble_CNN', 'RF', 'LGBM']
    groups = cpt_exhausive_dataset['ID'].values.flatten()
    cv = LeaveOneGroupOut()

    X, y = create_sequence_dataset()

    for m in methods:

        rf_scores = None; lgbm_scores = None

        if m == 'Ensemble_CNN':
            model = Collapse_CNN(latent_features=16, image_width=11)
            model.compile(optimizer='adam', loss='mse')
            preds = cross_val_predict(model, X, y, cv=cv, groups=groups, fit_params = NN_param_dict, n_jobs=-1)

            for dec in ['RF', 'LGBM']:
                encoder = model.cnn_encoder
                if dec == 'RF':
                    decoder = MultiOutputRegressor(RandomForestRegressor(**RF_param_dict))
                    rf_preds = cross_val_predict(decoder, encoder(X), y, cv=cv, groups=groups, n_jobs=-1)
                elif dec == 'LGBM':
                    decoder = MultiOutputRegressor(LGBMRegressor(**LGBM_param_dict))
                    lgbm_preds = cross_val_predict(decoder, encoder(X), y, cv=cv, groups=groups, n_jobs=-1)

            for label, pred in zip(['Ensemble_CNN', 'RF', 'LGBM'], [preds, rf_preds, lgbm_preds]):
                stds = []
                for k in range(pred.shape[-1]):
                    _, _, _, _, std, _ = evaluate_modeldist_norm(y[:, k], pred[:, k])
                    stds.append(std)
                



        else:
            if m == 'RF':
                decoder = MultiOutputRegressor(RandomForestRegressor(**RF_param_dict))
                
            elif m == 'LGBM':
                decoder = MultiOutputRegressor(LGBMRegressor(**LGBM_param_dict))

            preds = cross_val_predict(Collapse_tree(decoder), X, y, cv=cv, groups=groups, n_jobs=-1)
        

        
