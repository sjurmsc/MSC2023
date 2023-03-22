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

    X, y, groups = create_sequence_dataset(sequence_length=10,
                                           n_bootstraps = 20,
                                           add_noise=0.1,
                                           cumulative_seismic=False,
                                           random_flip=True,
                                           groupby='cpt_loc') # groupby can be 'cpt_loc' or 'borehole'

    # Shuffle training data
    X_train, y_train, groups_train = shuffle(X, y, groups, random_state=1)

    g_name_gen = give_modelname()
    gname, _ = next(g_name_gen)
    if not Path(f'../Models/{gname}').exists(): Path(f'../Models/{gname}').mkdir()

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

 
    rf_scores = None; lgbm_scores = None
    Histories = []

    for i, (train_index, test_index) in enumerate(cv.split(X_train, y_train, groups_train)):
        model, encoder = ensemble_CNN_model(n_members=1)
        if i==0: model.summary()

        print('Fold', i+1, 'of', cv.get_n_splits(groups=groups_train))
        X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
        y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
        groups_train_cv, groups_test_cv = groups_train[train_index], groups_train[test_index]

        History = model.fit(X_train_cv, y_train_cv, **NN_param_dict)
        Histories.append(History)

        encoder.save(f'../Models/{gname}/Ensemble_CNN_encoder_{i}.h5')
        model.save(f'../Models/{gname}/Ensemble_CNN_{i}.h5')

        # Plot the training and validation loss
        plot_history(History, filename=f'../Models/{gname}/Ensemble_CNN_{i}.png')

        # Adding predictions to a numpy array
        if i == 0:
            preds = model.predict(X_test_cv)
        else:
            preds = np.vstack((preds, model.predict(X_test_cv)))

        encoded_data = encoder.predict(X_test_cv)

        for dec in ['RF', 'LGBM']:
            if dec == 'RF':
                decoder = MultiOutputRegressor(RandomForestRegressor(**RF_param_dict))
                decoder.fit(encoded_data, y_train_cv)
                print('RF score:', decoder.score(encoder.predict(X_test_cv), y_test_cv))
                rf_preds = decoder.predict(encoder.predict(X_test_cv))
            elif dec == 'LGBM':
                decoder = MultiOutputRegressor(LGBMRegressor(**LGBM_param_dict))
                decoder.fit(encoded_data, y_train_cv)
                print('LGBM score:', decoder.score(encoder.predict(X_test_cv), y_test_cv))
                lgbm_preds = decoder.predict(encoder(X_test_cv))

    # Save the predictions
    np.save(f'../Models/{gname}/Ensemble_CNN_preds.npy', preds)

        

    for label, pred in zip(['Ensemble_CNN', 'RF', 'LGBM'], [preds, rf_preds, lgbm_preds]):
        stds = []
        for k in range(pred.shape[-1]):
            _, _, _, _, std, _ = evaluate_modeldist_norm(y[:, k], pred[:, k])
            stds.append(std)

    with open(f'../Models/{gname}/std_results.txt', 'a') as f:
        f.write(f'{label} stds: {stds}')


        

        
