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
from sklearn.model_selection import LeaveOneGroupOut, cross_validate, cross_val_predict, train_test_split
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.manifold import TSNE
from sklearn.multioutput import MultiOutputRegressor
from keras.losses import mean_squared_error

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



if __name__ == '__main__':

    # Retrieve data
    cpt_exhausive_dataset = read_csv(r'../OneDrive - NGI/Documents/NTNU/MSC_DATA/Database.csv')


    # Creating the model
    methods = ['Ensemble_CNN'] #, 'RF', 'LGBM']
    cv = LeaveOneGroupOut()

    X, y, groups = create_sequence_dataset(n_bootstraps = 20, groupby='cpt_loc')

    # Split data into training and test set
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(X, y, groups, test_size=0.2, random_state=1, stratify=groups)

    NN_param_dict = {
        'epochs': 200,
        'batch_size': 20,
        'validation_data': (X_test, y_test)

        }

    for m in methods:

        rf_scores = None; lgbm_scores = None

        if m == 'Ensemble_CNN':
            model = Collapse_CNN(latent_features=16, image_width=11)
            model.compile(optimizer='adam', loss=mean_squared_error, metrics=['mse'])

            for i, (train_index, test_index) in enumerate(cv.split(X_train, y_train, groups_train)):
                X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
                y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
                groups_train_cv, groups_test_cv = groups_train[train_index], groups_train[test_index]

                model.fit(X_train_cv, y_train_cv, **NN_param_dict)

                if i == 0:
                    preds = model.predict(X_train)
                else:
                    preds = np.vstack((preds, model.predict(X_train)))

            
            # preds = cross_val_predict(model, X_train, y_train, cv=cv, groups=groups_train, fit_params = NN_param_dict, n_jobs=-1)

            for dec in ['RF', 'LGBM']:
                encoder = model.cnn_encoder
                if dec == 'RF':
                    decoder = MultiOutputRegressor(RandomForestRegressor(**RF_param_dict))
                    rf_preds = cross_val_predict(decoder, encoder(X), y, cv=cv, groups=groups, n_jobs=-1)
                elif dec == 'LGBM':
                    decoder = MultiOutputRegressor(LGBMRegressor(**LGBM_param_dict))
                    lgbm_preds = cross_val_predict(decoder, encoder(X), y, cv=cv, groups=groups, n_jobs=-1)

            for label, pred in zip(['Ensemble_CNN'], [preds]): # zip(['Ensemble_CNN', 'RF', 'LGBM'], [preds, rf_preds, lgbm_preds]):
                stds = []
                for k in range(pred.shape[-1]):
                    _, _, _, _, std, _ = evaluate_modeldist_norm(y[:, k], pred[:, k])
                    stds.append(std)
            Print(f'{label} stds: {stds}')
                



        else:
            if m == 'RF':
                decoder = MultiOutputRegressor(RandomForestRegressor(**RF_param_dict))
                
            elif m == 'LGBM':
                decoder = MultiOutputRegressor(LGBMRegressor(**LGBM_param_dict))

            preds = cross_val_predict(Collapse_tree(decoder), X, y, cv=cv, groups=groups, n_jobs=-1)
        

        
