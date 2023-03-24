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
import json
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict, train_test_split
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.manifold import TSNE
from sklearn.multioutput import MultiOutputRegressor
# from tensorflow.compat.v1.keras.utils import plot_model

# My scripts
from Log import *
from Architectures import *
from Feat_aug import *
from NGI.GM_Toolbox import evaluate_modeldist_norm

from time import time




if __name__ == '__main__':
    # Setting up the cross validation
    cv = LeaveOneGroupOut()

    # Getting scaler for the data
    scaler = get_cpt_data_scaler()

    dataset_params = {
        'n_neighboring_traces'  : 5,
        'zrange'                : (30, 100),
        'n_bootstraps'          : 10,
        'add_noise'             : 0.01,
        'max_distance_to_cdp'   : 20,
        'cumulative_seismic'    : False,
        'random_flip'           : True,
        'random_state'          : 1,
        'groupby'               : 'cpt_loc',
        'y_scaler'              : scaler
        }

    X_train, y_train, groups_train = create_sequence_dataset(sequence_length=10,
                                                             stride=1,
                                                             **dataset_params) # groupby can be 'cpt_loc' or 'borehole'

    full_trace = create_full_trace_dataset(**dataset_params)
    X_full, y_full, groups_full, full_nan_idx, full_no_nan_idx, sw_idxs, extr_idxs = full_trace
    del full_trace

    GGM = array(sw_idxs) + 1

    g_name_gen = give_modelname()
    gname, _ = next(g_name_gen)
    if not Path(f'./Models/{gname}').exists(): Path(f'./Models/{gname}').mkdir()

    describe_data(X_train, y_train, groups_train, mdir=f'./Models/{gname}/')

    # Configurations for models
    RF_param_dict = {
        'max_depth'         : 20,
        'n_estimators'      : 20,
        'min_samples_leaf'  : 1,
        'min_samples_split' : 4,
        'bootstrap'         : True,
        'criterion'         : 'mse',
        'n_jobs'            : -1
        }

    LGBM_param_dict = {
        'num_leaves'        : 31,
        'max_depth'         : 20,
        'n_estimators'      : 100,
        'n_jobs'            : -1
        }

    NN_param_dict = {
        'epochs'            : 2,
        'batch_size'        : 25
        }

    # Training time dict
    training_time_dict = {}
 
    rf_scores = None; lgbm_scores = None
    Histories = []

    for i, (train_index, test_index) in enumerate(cv.split(X_train, y_train, groups_train)):
        Train_groups    = np.unique(groups_train[train_index])
        Test_group      = np.unique(groups_train[test_index])

        # Getting the indices of the train and test data for the current cv split
        in_train = np.isin(groups_full, Train_groups)
        in_test = np.isin(groups_full, Test_group)

        # Creating full trace cv dataset
        X_train_full    = X_full[in_train]
        y_train_full    = y_full[in_train]
        X_test_full     = X_full[in_test]
        y_test_full     = y_full[in_test]

        # Getting the indices of the nan values for coloring plots
        full_nan_idx_train = full_nan_idx[in_train]
        full_nan_idx_test = full_nan_idx[in_test]

        # Getting the indices of the non-nan values for training trees
        full_no_nan_idx_train = full_no_nan_idx[in_train]
        full_no_nan_idx_test = full_no_nan_idx[in_test]

        # Setting up the model
        image_width = 2*dataset_params['n_neighboring_traces'] + 1
        learning_rate = 0.001
        model, encoder = ensemble_CNN_model(n_members=1, 
                                            image_width=image_width, 
                                            learning_rate=learning_rate,
                                            enc = 'cnn',
                                            dec = 'lstm')
        if i==0: model.summary()

        # Preparing the data to train the CNN
        print('Fold', i+1, 'of', cv.get_n_splits(groups=groups_train))
        X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
        y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
        groups_train_cv, groups_test_cv = groups_train[train_index], groups_train[test_index]

        # Timing the model
        training_time_dict[i] = {}
        t0 = time()

        # Training the model
        History = model.fit(X_train_cv, y_train_cv, **NN_param_dict)
        training_time_dict[i]['CNN'] = time() - t0

        Histories.append(History)

        encoder.save(f'./Models/{gname}/Fold{i+1}/Ensemble_CNN_encoder_{i}.h5')
        model.save(f'./Models/{gname}/Fold{i+1}/Ensemble_CNN_{i}.h5')

        # Plot the training and validation loss
        plot_history(History, filename=f'./Models/{gname}/Fold{i+1}/Ensemble_CNN_{i}.png')

        # Plotting the models
        # plot_model(model, to_file=f'./Models/{gname}/Ensemble_CNN_{i}.png', show_shapes=True, show_layer_names=True)
        # plot_model(encoder, to_file=f'./Models/{gname}/Ensemble_CNN_encoder_{i}.png', show_shapes=True, show_layer_names=True)

        # Adding predictions to a numpy array
        if i == 0:
            trues = y_test_cv
            preds = model.predict(X_test_cv)
        else:
            trues = np.vstack((trues, y_test_cv))
            preds = np.vstack((preds, model.predict(X_test_cv)))

        encoded_data = encoder(X_train_full).numpy()
        tree_train_input_shape = (-1, encoded_data.shape[-1])
        idx_train = full_no_nan_idx_train.flatten()
        idx_nan_train = full_nan_idx_train.flatten()
        encoded_data = encoded_data.reshape(tree_train_input_shape)
        flat_y_train = y_train_full.reshape(-1, y_train_full.shape[-1])
        
        
        test_prediction = encoder(X_test_full).numpy()
        tree_test_input_shape = (-1, test_prediction.shape[-1])
        idx_test = full_no_nan_idx_test.flatten()
        idx_nan_test = full_nan_idx_test.flatten()
        test_prediction = test_prediction.reshape(tree_test_input_shape)
        flat_y_test = y_test_full.reshape(-1, y_test_full.shape[-1])

        t_train_pred = encoded_data[idx_train]
        t_flat_y_train = flat_y_train[idx_train]

        t_test_pred = test_prediction[idx_test]
        t_flat_y = flat_y_test[idx_test]

        for dec in ['RF', 'LGBM']:
            if dec == 'RF':
                print('Fitting RF')

                t0 = time()
                rf_decoder = MultiOutputRegressor(RandomForestRegressor(**RF_param_dict))
                rf_decoder.fit(t_train_pred, t_flat_y_train)
                training_time_dict[i]['RF'] = time() - t0
                
                s = rf_decoder.score(t_test_pred, t_flat_y)
                print('RF score: {}'.format(s))
                rf_preds = predict_encoded_tree(encoder, rf_decoder, X_test_cv)
    
            elif dec == 'LGBM':
                print('Fitting LGBM')
                t0 = time()
                lgbm_decoder = MultiOutputRegressor(LGBMRegressor(**LGBM_param_dict))
                lgbm_decoder.fit(t_train_pred, t_flat_y_train)
                training_time_dict[i]['LGBM'] = time() - t0

                s = lgbm_decoder.score(t_test_pred, t_flat_y)
                print('LGBM score: {}'.format(s))
                lgbm_preds = predict_encoded_tree(encoder, lgbm_decoder, X_test_cv)

        # Plotting the predictions
        for m, title in zip([model, [encoder, rf_decoder], [encoder, lgbm_decoder]], ['CNN', 'RF', 'LGBM']):
            create_loo_trace_prediction(m, 
                                        X_test_full, 
                                        y_test_full, 
                                        zrange=dataset_params['zrange'],
                                        filename=f'./Models/{gname}/Fold{i+1}/Ensemble_CNN_{title}_{i}.png',
                                        title=title)
        
        # Plotting the latent space
        plot_latent_space(encoder, 
                          X_test_full, 
                          full_no_nan_idx_test, 
                          full_nan_idx_test, 
                          GGM,
                          filename=f'./Models/{gname}/Fold{i+1}/Ensemble_CNN_latent_space_{i}.png')
        
        


    # Save the training times
    with open(f'./Models/{gname}/training_times.txt', 'w') as f:
        f.write(json.dumps(training_time_dict))

    # Save the predictions
    np.save(f'./Models/{gname}/Ensemble_CNN_preds.npy', preds)

    # plot the latent space, colored by structural model
    # plot_latent_space(encoder, X_t, y_train, groups_train, filename=f'./Models/{gname}/Ensemble_CNN_latent_space.png')

    for i in range(trues.shape[0]):
        trues[i] = scaler.inverse_transform(trues[i])

    for label, pred in zip(['Ensemble_CNN', 'RF', 'LGBM'], [preds, rf_preds, lgbm_preds]):
        stds = []
        print('Evaluating model stds for {}'.format(label))

        trans_pred = np.zeros(pred.shape)
        # Inverse transform the data
        for i in range(pred.shape[0]):
            trans_pred[i] = scaler.inverse_transform(pred[i])

        for k in range(pred.shape[-1]):
            _, _, _, _, std, _ = evaluate_modeldist_norm(trues[:, :, k].flatten(), trans_pred[:, :, k].flatten())
            stds.append(std)
        print('std for {} is: {}'.format(label, stds))

    # with open(f'./Models/{gname}/std_results.txt', 'a') as f:
    #     f.write(f'{label} stds: {stds}')

    # Create prediction crossplots



        

        
