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

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict, train_test_split
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.manifold import TSNE
from sklearn.multioutput import MultiOutputRegressor

# My scripts
from Log import *
from Architectures import *
from Feat_aug import *
from NGI.GM_Toolbox import evaluate_modeldist_norm





if __name__ == '__main__':

    # Retrieve data
    # cpt_exhausive_dataset = read_csv(r'../OneDrive - NGI/Documents/NTNU/MSC_DATA/Database.csv')


    # Creating the model

    cv = LeaveOneGroupOut()

    X, y, groups = create_sequence_dataset(n_bootstraps = 20, groupby='cpt_loc') # groupby can be 'cpt_loc' or 'borehole'

    # Shuffle training data
    X_train, y_train, groups_train = shuffle(X, y, groups, random_state=1)

    # X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(X, y, groups, random_state=1, stratify=groups) # Stratify makes it so that the test set has the same distribution of classes as the training set


    # Configurations for models
    RF_param_dict = {
        'max_depth'         : 20,
        'n_estimators'      : 20,
        'min_samples_leaf'  : 1,
        'min_samples_split' : 4,
        'bootstrap'         : True,
        'criterion'         : 'mse'
        }

    LGBM_param_dict = {
        'num_leaves'        : 31,
        'max_depth'         : 20,
        'n_estimators'      : 100
        }

    NN_param_dict = {
        'epochs'            : 5,
        'batch_size'        : 85
        }

 
    rf_scores = None; lgbm_scores = None
    Histories = []

    model = ensemble_CNN_model(n_members=1)
    model.summary()

    for i, (train_index, test_index) in enumerate(cv.split(X_train, y_train, groups_train)):
        print('Fold', i+1, 'of', cv.get_n_splits())
        X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
        y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
        groups_train_cv, groups_test_cv = groups_train[train_index], groups_train[test_index]

        History = model.fit(X_train_cv, y_train_cv, **NN_param_dict)
        Histories.append(History)

        model.save(f'../Models/Ensemble_CNN_{i}.h5')

        # Plot the training and validation loss
        plot_history(History, val=True, filename=f'../Models/Ensemble_CNN_{i}.png')

        # Adding predictions to a numpy array
        if i == 0:
            preds = model.predict(X_test_cv)
        else:
            preds = np.vstack((preds, model.predict(X_test_cv)))

    # Save the predictions
    np.save('../Models/Ensemble_CNN_preds.npy', preds)

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

    with open('results.txt', 'a') as f:
        f.write(f'{label} stds: {stds}')


        

        
