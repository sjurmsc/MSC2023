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

from time import time




if __name__ == '__main__':

    # Retrieve data
    # cpt_exhausive_dataset = read_csv(r'../OneDrive - NGI/Documents/NTNU/MSC_DATA/Database.csv')


    # Creating the model

    cv = LeaveOneGroupOut()

    X, y, groups = create_sequence_dataset(sequence_length=10,
                                           n_bootstraps = 20,
                                           add_noise=0.1,
                                           max_distance_to_cdp=10,
                                           cumulative_seismic=False,
                                           random_flip=True,
                                           groupby='cpt_loc') # groupby can be 'cpt_loc' or 'borehole'

    # Shuffle training data
    X_train, y_train, groups_train = shuffle(X, y, groups, random_state=1)

    g_name_gen = give_modelname()
    gname, _ = next(g_name_gen)
    if not Path(f'./Models/{gname}').exists(): Path(f'./Models/{gname}').mkdir()

    describe_data(X, y, groups, mdir=f'./Models/{gname}/')

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
        'batch_size'        : 88
        }

    # Training time dict
    training_time_dict = {}
 
    rf_scores = None; lgbm_scores = None
    Histories = []

    for i, (train_index, test_index) in enumerate(cv.split(X_train, y_train, groups_train)):
        model, encoder = ensemble_CNN_model(n_members=1)
        if i==0: model.summary()

        print('Fold', i+1, 'of', cv.get_n_splits(groups=groups_train))
        X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
        y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
        groups_train_cv, groups_test_cv = groups_train[train_index], groups_train[test_index]

        # Training the model
        t0 = time()
        History = model.fit(X_train_cv, y_train_cv, **NN_param_dict)

        training_time_dict[i] = {}
        training_time_dict[i]['CNN'] = time() - t0
        Histories.append(History)

        encoder.save(f'./Models/{gname}/Ensemble_CNN_encoder_{i}.h5')
        model.save(f'./Models/{gname}/Ensemble_CNN_{i}.h5')

        # Plot the training and validation loss
        plot_history(History, filename=f'./Models/{gname}/Ensemble_CNN_{i}.png')

        # Adding predictions to a numpy array
        if i == 0:
            preds = model.predict(X_test_cv)
        else:
            preds = np.vstack((preds, model.predict(X_test_cv)))

        encoded_data = encoder.predict(X_train_cv)[:, 0, :, :]
        tree_train_input_shape = (encoded_data.shape[0]*encoded_data.shape[1], 16)
        encoded_data = encoded_data.reshape(tree_train_input_shape)
        flat_y_train = y_train_cv.reshape(y_train_cv.shape[0]*y_train_cv.shape[1], 3)
        
        
        test_prediction = encoder.predict(X_test_cv)[:, 0, :, :]
        tree_test_input_shape = (test_prediction.shape[0]*test_prediction.shape[1], 16)
        print(tree_test_input_shape)
        test_prediciton = test_prediction.reshape(tree_test_input_shape)
        flat_y_test = y_test_cv.reshape(y_test_cv.shape[0]*y_test_cv.shape[1], 3)

        print(encoded_data.shape, flat_y_train.shape, test_prediction.shape, flat_y_test.shape)

        for dec in ['RF', 'LGBM']:
            if dec == 'RF':
                print('Fitting RF')

                t0 = time()
                decoder = MultiOutputRegressor(RandomForestRegressor(**RF_param_dict), n_jobs=-1)
                decoder.fit(encoded_data, flat_y_train)
                training_time_dict[i]['RF'] = time() - t0
                
                print('RF score:', decoder.score(test_prediction, flat_y_test))
                rf_preds = decoder.predict(test_prediction)
    
            elif dec == 'LGBM':
                print('Fitting LGBM')
                t0 = time()
                decoder = MultiOutputRegressor(LGBMRegressor(**LGBM_param_dict), n_jobs=-1)
                decoder.fit(encoded_data, flat_y_train)
                training_time_dict[i]['LGBM'] = time() - t0

                print('LGBM score:', decoder.score(test_prediction, flat_y_test))
                lgbm_preds = decoder.predict(test_prediction)

    # Save the predictions
    np.save(f'./Models/{gname}/Ensemble_CNN_preds.npy', preds)

        

    for label, pred in zip(['Ensemble_CNN', 'RF', 'LGBM'], [preds, rf_preds, lgbm_preds]):
        stds = []
        print('Evaluating model stds for {}'.format(label))
        for k in range(pred.shape[-1]):
            _, _, _, _, std, _ = evaluate_modeldist_norm(y[:, k], pred[:, k])
            stds.append(std)

    with open(f'./Models/{gname}/std_results.txt', 'a') as f:
        f.write(f'{label} stds: {stds}')

    # Create prediction crossplots



        

        
