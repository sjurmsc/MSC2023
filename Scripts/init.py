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
import json
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict, train_test_split
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.manifold import TSNE
from sklearn.multioutput import MultiOutputRegressor
import tensorflow as tf
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
        'zrange'                : (35, 100),
        'n_bootstraps'          : 5,
        'add_noise'             : False,
        'max_distance_to_cdp'   : 25,
        'cumulative_seismic'    : False,
        'random_flip'           : False,
        'random_state'          : 1,
        'groupby'               : 'cpt_loc',
        'y_scaler'              : 'minmax',
        'exclude_BH'            : True
        }

    g_name_gen = give_modelname()
    gname, _ = next(g_name_gen)
    if not Path(f'./Models/{gname}').exists(): Path(f'./Models/{gname}').mkdir()

    full_trace = create_full_trace_dataset(**dataset_params, ydata='mmm')
    X_full, y_full, groups_full, full_nan_idx, full_no_nan_idx, sw_idxs, extr_idxs, GGM, GGM_unc, minmax_full = full_trace
    del full_trace

    dataset_params['sequence_length'] = 100
    dataset_params['stride'] = 1

    X_train, y_train, groups_train, z_train, GGM_train, GGM_unc_train = create_sequence_dataset(**dataset_params)

    describe_data(X_train, y_train, groups_train, GGM_train, mdir=f'./Models/{gname}/')

    # Configurations for models
    RF_param_dict = {
        'max_depth'         : 20,
        'n_estimators'      : 300,
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
        'epochs'            : 1000,
        'batch_size'        : 30
        }
    
    # Save all parameter dictionaries to a json file
    with open(f'./Models/{gname}/param_dict.json', 'w') as f:
        json.dump({'dataset' : dataset_params,'RF' : RF_param_dict, 'LGBM' : LGBM_param_dict, 'NN' : NN_param_dict}, f, indent=4)
    
    encoder_type = 'cnn'
    decoder_type = 'ann'
    n_members    = 1
    latent_features = 16

    # Training time dict
    training_time_dict = {}
 
    rf_scores = None; lgbm_scores = None
    Histories = []

    cols = ['Depth', 'GGM', 'GGM_uncertainty', 'True_qc', 'CNN_qc', 'RF_qc', 'LGBM_qc', 'True_fs', 'CNN_fs', 'RF_fs', 'LGBM_fs', 'True_u2', 'CNN_u2', 'RF_u2', 'LGBM_u2']

    COMPARE_df = pd.DataFrame(columns=cols)
    loss_dict = {}

    for i, (train_index, test_index) in enumerate(cv.split(X_train, y_train, groups_train)):
        Train_groups    = np.unique(groups_train[train_index])
        Test_group      = np.unique(groups_train[test_index])

        # Getting the indices of the train and test data for the current cv split
        in_train = np.isin(groups_full, Train_groups)
        in_test = np.isin(groups_full, Test_group)

        # Creating full trace cv dataset
        X_train_full    = X_full[in_train].copy()
        y_train_full    = y_full[in_train].copy()
        X_test_full     = X_full[in_test].copy()
        y_test_full     = y_full[in_test].copy()
        minmax_test_full = (minmax_full[0][in_test].copy(), minmax_full[1][in_test].copy())
        GGM_test_full   = GGM[in_test].copy()

        # Get the unique GGMs that are in the test set but not in the train set
        GGM_test_not_in_train = np.setdiff1d(np.unique(GGM_train[test_index]), np.unique(GGM_train[train_index]))
        if len(GGM_test_not_in_train) > 0:
            tempdir = f'./Models/{gname}/Fold{i+1}'
            if not Path(tempdir).exists(): Path(tempdir).mkdir()
            with open(f'./Models/{gname}/Fold{i+1}/GGM_test_not_in_train.txt', 'w') as f:
                f.write(str(GGM_test_not_in_train))

        # Getting the indices of the nan values for coloring plots
        full_nan_idx_train = full_nan_idx[in_train].copy()
        full_nan_idx_test = full_nan_idx[in_test].copy()

        # Getting the indices of the non-nan values for training trees
        full_no_nan_idx_train = full_no_nan_idx[in_train]
        full_no_nan_idx_test = full_no_nan_idx[in_test]

        # Setting up the model
        image_width = 2*dataset_params['n_neighboring_traces'] + 1
        learning_rate = 0.001
        model, encoder, model_mean = ensemble_CNN_model(n_members=n_members,
                                                        latent_features=latent_features, 
                                                        image_width=image_width, 
                                                        learning_rate=learning_rate,
                                                        enc = encoder_type,
                                                        dec = decoder_type,
                                                        reconstruct=True)
        if i==0: model.summary()

        # Preparing the data to train the CNN
        print('Fold', i+1, 'of', cv.get_n_splits(groups=groups_train))
        X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
        y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
        GGM_train_cv, GGM_test_cv = GGM_train[train_index], GGM_train[test_index]
        groups_train_cv, groups_test_cv = groups_train[train_index], groups_train[test_index]

        # Timing the model
        training_time_dict[i] = {}
        t0 = time()

        # Training the model
        History = model.fit(X_train_cv, [y_train_cv, X_train_cv], **NN_param_dict)
        training_time_dict[i]['CNN'] = time() - t0

        Histories.append(History)

        # Add loss to loss_dict
        loss_dict[f'Fold{i+1}'] = History.history['loss']

        encoder.save(f'./Models/{gname}/Fold{i+1}/Ensemble_CNN_encoder_{i}.h5')
        model_mean.save(f'./Models/{gname}/Fold{i+1}/Ensemble_CNN_{i}.h5')
        model.save(f'./Models/{gname}/Fold{i+1}/Ensemble_CNN_w_reconstruct_{i}.h5')

        with open(f'./Models/{gname}/Fold{i+1}/LOO_group.txt', 'w') as f:
            f.write(f'LOO group: {Test_group}')

        # Plot the training and validation loss
        plot_history(History, filename=f'./Models/{gname}/Fold{i+1}/Ensemble_CNN_{i}.png')

        # Encoder and model summary to text file
        with open(f'./Models/{gname}/Fold{i+1}/Ensemble_CNN_encoder_{i}.txt', 'w') as f:
            encoder.summary(print_fn=lambda x: f.write(x + '\n'))
        with open(f'./Models/{gname}/Fold{i+1}/Ensemble_CNN_{i}.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))


        # Adding predictions to a numpy array
        if i == 0:
            trues = y_test_cv
            preds = model_mean.predict(X_test_cv)
            z = z_train[test_index]
            ggms = GGM_test_cv
        else:
            trues = np.vstack((trues, y_test_cv))
            preds = np.vstack((preds, model_mean.predict(X_test_cv)))
            z = np.vstack((z, z_train[test_index]))
            ggms = np.vstack((ggms, GGM_test_cv))

        # encoded_data = encoder(X_train_full).numpy()
        encoded_data = encoder(X_train_cv).numpy()
        tree_train_input_shape = (-1, encoded_data.shape[-1])
        # idx_train = full_no_nan_idx_train.flatten()
        # idx_nan_train = full_nan_idx_train.flatten()
        encoded_data = encoded_data.reshape(tree_train_input_shape)
        # flat_y_train = y_train_full.reshape(-1, y_train_full.shape[-1])
        flat_y_train = y_train_cv.reshape(-1, y_train_cv.shape[-1])
        
        
        # test_prediction = encoder(X_test_full).numpy()
        test_prediction = encoder(X_test_cv).numpy()
        tree_test_input_shape = (-1, test_prediction.shape[-1])
        # idx_test = full_no_nan_idx_test.flatten()
        # idx_nan_test = full_nan_idx_test.flatten()
        test_prediction = test_prediction.reshape(tree_test_input_shape)
        # flat_y_test = y_test_full.reshape(-1, y_test_full.shape[-1])
        flat_y_test = y_test_cv.reshape(-1, y_test_cv.shape[-1])

        # t_train_pred = encoded_data[idx_train]
        # t_flat_y_train = flat_y_train[idx_train]

        # Temporary
        t_train_pred = encoded_data
        t_flat_y_train = flat_y_train

        # t_test_pred = test_prediction[idx_test]
        # t_flat_y = flat_y_test[idx_test]

        # Temporary
        t_test_pred = test_prediction
        t_flat_y = flat_y_test

        tree_scores = {}

        for dec in ['RF', 'LGBM']:
            if dec == 'RF':
                print('Fitting RF')

                t0 = time()
                rf_decoder = MultiOutputRegressor(RandomForestRegressor(**RF_param_dict))
                rf_decoder.fit(t_train_pred, t_flat_y_train)
                training_time_dict[i]['RF'] = time() - t0
                
                s = rf_decoder.score(t_test_pred, t_flat_y)
                print('RF score: {}'.format(s))
                tree_scores['RF'] = s
                rf = predict_encoded_tree(encoder, rf_decoder, X_test_cv)
                if i == 0:
                    rf_preds = rf
                else:
                    rf_preds = np.vstack((rf_preds, rf))
    
            elif dec == 'LGBM':
                print('Fitting LGBM')
                t0 = time()
                lgbm_decoder = MultiOutputRegressor(LGBMRegressor(**LGBM_param_dict))
                lgbm_decoder.fit(t_train_pred, t_flat_y_train)
                training_time_dict[i]['LGBM'] = time() - t0

                s = lgbm_decoder.score(t_test_pred, t_flat_y)
                print('LGBM score: {}'.format(s))
                tree_scores['LGBM'] = s
                lgbm = predict_encoded_tree(encoder, lgbm_decoder, X_test_cv)
                if i == 0:
                    lgbm_preds = lgbm
                else:
                    lgbm_preds = np.vstack((lgbm_preds, lgbm))
        
        # export the tree scores
        with open(f'./Models/{gname}/Fold{i+1}/tree_scores.json', 'w') as f:
            json.dump(tree_scores, f, indent=4)

        # Plotting the predictions
        for m, title in zip([model_mean, [encoder, rf_decoder], [encoder, lgbm_decoder]], [decoder_type.upper(), 'RF', 'LGBM']):
            create_loo_trace_prediction(m, 
                                        X_test_full, 
                                        y_test_full, 
                                        zrange=dataset_params['zrange'],
                                        filename=f'./Models/{gname}/Fold{i+1}/Ensemble_CNN_{title}_{i}.png',
                                        title=title,
                                        minmax=minmax_test_full,
                                        scale=True)
            prediction_scatter_plot(m,
                                    X_test_full,
                                    y_test_full,
                                    filename=f'./Models/{gname}/Fold{i+1}/Ensemble_CNN_{title}_scatter_{i}.png',
                                    title=title,
                                    scale=True)
        
        # Plotting the latent space
        plot_latent_space(encoder,
                          latent_features, 
                          X_test_full, 
                          full_no_nan_idx_test, 
                          full_nan_idx_test, 
                          GGM_test_full,
                          filename=f'./Models/{gname}/Fold{i+1}/Ensemble_CNN_latent_space_{i}.png')
    
    # Inverse scaling
    for ii in range(trues.shape[0]):
        trues[ii, :, :] = scaler.inverse_transform(trues[ii, :, :])
        preds[ii, :, :] = scaler.inverse_transform(preds[ii, :, :])
        rf_preds[ii, :, :] = scaler.inverse_transform(rf_preds[ii, :, :])
        lgbm_preds[ii, :, :] = scaler.inverse_transform(lgbm_preds[ii, :, :])

    # Adding predictions to the compare dataframe
    comp = {'Depth'             : z.flatten(),
            'GGM'               : ggms.flatten(),
            'GGM_uncertainty'   : np.zeros_like(ggms.flatten()),
            'True_qc'           : trues[:, :, 0].flatten(),
            'CNN_qc'            : preds[:, :, 0].flatten(),
            'RF_qc'             : rf_preds[:, :, 0].flatten(),
            'LGBM_qc'           : lgbm_preds[:, :, 0].flatten(),
            'True_fs'           : trues[:, :, 1].flatten(),
            'CNN_fs'            : preds[:, :, 1].flatten(),
            'RF_fs'             : rf_preds[:, :, 1].flatten(),
            'LGBM_fs'           : lgbm_preds[:, :, 1].flatten(),
            'True_u2'           : trues[:, :, 2].flatten(),
            'CNN_u2'            : preds[:, :, 2].flatten(),
            'RF_u2'             : rf_preds[:, :, 2].flatten(),
            'LGBM_u2'           : lgbm_preds[:, :, 2].flatten(),
            }
    
    COMPARE_df = pd.DataFrame(comp)

    # Save the losses of the different folds to a dict
    with open(f'./Models/{gname}/loss_dict.json', 'w') as f:
        json.dump(loss_dict, f, indent=4)
    
    pred_dir = f'./Models/{gname}/Predictions/'
    if not Path(pred_dir).exists():
        Path(pred_dir).mkdir(parents=True)
    describe_data(X_train, preds, groups_train, ggms, mdir=pred_dir)

    # Save the predictions for each to a pickle
    with open(f'./Models/{gname}/Ensemble_CNN_preds.pkl', 'wb') as f:
        pickle.dump(COMPARE_df, f)

    # Calculate statisics, grouped by ggm unit
    filename = f'./Models/{gname}/Ensemble_CNN_stats.xlsx'
    make_cv_excel(filename, COMPARE_df)

    # Save the training times
    with open(f'./Models/{gname}/training_times.txt', 'w') as f:
        f.write(json.dumps(training_time_dict))

    




        

        
