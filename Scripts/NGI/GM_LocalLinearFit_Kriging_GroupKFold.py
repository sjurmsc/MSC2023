# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 22:20:24 2021

Kriging of slope and intercept per unit

@author: GuS
"""

#%% Import libraries
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.model_selection import cross_validate, cross_val_predict

from pykrige.rk import Krige
from pykrige.uk import UniversalKriging
from pykrige.ok import OrdinaryKriging

#%% Define functions
def load_obj(path_obj):
    with open(path_obj, 'rb') as f:
        return pickle.load(f)
    
def save_obj(path_obj, obj):
    with open(path_obj, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

#%% Load database
path_database = '../../09-Results/Stage-01/Database.csv'
df = pd.read_csv(path_database)

#%% Load Local linear regression results
path_dict = '../../09-Results/Stage-01/LocalLinearFit-Kfold-results_bounds.pkl'
df_lreg = load_obj(path_dict)

# list of units
unitlist = df['unit_geo'].dropna().unique()

#%% Kfold Cross validation Kriging of slope and intercept from linear regression
# param_dict = {
#     'method': 'universal',
#     'variogram_model': 'linear',
#     'nlags': 40,
#     'weight': True,
#     'drift_terms': 'regional_linear',
#     'pseudo_inv': True,
#     'pseudo_inv_type': 'pinv'
#     }
# param_dict_K = {
#     'variogram_model': 'linear',
#     'nlags': 40,
#     'weight': True,
#     'drift_terms': 'regional_linear',
#     'enable_plotting': False,
#     'pseudo_inv': True,
#     'pseudo_inv_type': 'pinv'
#     }

param_dict = {
    'method': 'universal',
    'variogram_model': 'spherical',
    'nlags': 40,
    'weight': True,
    'pseudo_inv': True,
    }
param_dict_K = {
    'variogram_model': 'spherical',
    'nlags': 40,
    'weight': True,
    'enable_plotting': False,
    'pseudo_inv': True,
    }

# Loop over features
df_CV_score = pd.DataFrame([])
df_CV_pred = pd.DataFrame([])
for feature in ['qc', 'fs', 'u2']:
    print('Data: ', feature)
    # Loop over units
    for unit in unitlist:
        print('\tUnit: ', unit)
        df_tmp_score = pd.DataFrame([])
        df_tmp_pred = pd.DataFrame([])
        X = df_lreg.loc[(df_lreg['feature']==feature) & (df_lreg['unit_geo']==unit), ['x','y']].values.astype(float)
        loc = df_lreg.loc[(df_lreg['feature']==feature) & (df_lreg['unit_geo']==unit), ['ID']].values.astype(int)
        aa = df_lreg.loc[(df_lreg['feature']==feature) & (df_lreg['unit_geo']==unit), 'slope']
        aa = np.array([x for x in aa]).flatten()
        bb = df_lreg.loc[(df_lreg['feature']==feature) & (df_lreg['unit_geo']==unit), 'intercept']
        bb = np.array([x for x in bb]).flatten()
        
        # Krige slope/interecpt on full dataset
        df_tmp_pred['ID'] = loc.tolist()
        df_tmp_pred['unit'] = unit
        df_tmp_pred['feature'] = feature
        if np.shape(X)[0]>1:            
            # UK_intercept = OrdinaryKriging(X[:,0], X[:,1], bb, **param_dict_K)
            # UK_slope = OrdinaryKriging(X[:,0], X[:,1], aa, **param_dict_K)
            UK_intercept = UniversalKriging(X[:,0], X[:,1], bb, **param_dict_K)
            UK_slope = UniversalKriging(X[:,0], X[:,1], aa, **param_dict_K) 
            
            # if np.sum(bb) == 0:
            #     UK_intercept = 0
            # else:
            #     UK_intercept = UniversalKriging(X[:,0], X[:,1], bb, **param_dict_K)
            # if np.abs(np.sum(np.diff(aa)))<1e-6:
            #     UK_slope = np.mean(aa)
            # else:
            #     UK_slope = UniversalKriging(X[:,0], X[:,1], aa, **param_dict_K) 
               
            df_tmp_score.loc[0,'slope_kri_obj'] = UK_slope
            df_tmp_score.loc[0,'intercept_kri_obj'] = UK_intercept
        
        # LeaveOneGroupOut and GroupKfold Cross validation kriging
        groups = loc.flatten()
        # LeaveOneGroupout Cross validated predition
        cv = LeaveOneGroupOut()
        if np.abs(np.sum(np.diff(aa)))<1e-6:
            df_tmp_pred['slope'] = aa[0]
        else:
            df_tmp_pred['slope'] = cross_val_predict(Krige(**param_dict),
                                                  X, aa, cv=cv, groups=groups,
                                                  n_jobs=-1, verbose=0)
        if np.sum(bb) == 0:
            df_tmp_pred['intercept'] = 0
        else:
            df_tmp_pred['intercept'] = cross_val_predict(Krige(**param_dict),
                                                          X, bb, cv=cv, groups=groups,
                                                          n_jobs=-1, verbose=0)
        
        df_CV_pred = df_CV_pred.append(df_tmp_pred, ignore_index=True)
        
        n_splits = np.min([5, loc.shape[0]])
        # print('\tUnit: ', unit, n_splits>2)
        # if n_splits > 2:
        print('\tUnit: ', unit, n_splits>0)
        if n_splits > 0:
            # GroupKfold Cross validation scoring
            # cv = GroupKFold(n_splits=n_splits)
            cv = LeaveOneGroupOut()
            df_tmp_score.loc[0,'unit'] = unit
            if np.abs(np.sum(np.diff(aa)))<1e-6:
                df_tmp_score.loc[0,'slope_r2'] = 1
                df_tmp_score.loc[0,'slope_mae'] = 0
                df_tmp_score.loc[0,'slope_r2_std'] = 0
                df_tmp_score.loc[0,'slope_mae_std'] = 0
            else:
                score_slope = cross_validate(Krige(**param_dict),
                                             X, aa,
                                             scoring=['r2', 'neg_mean_absolute_error'], 
                                             cv=cv, groups=groups,
                                             n_jobs=-1, verbose=0)
                df_tmp_score.loc[0,'slope_r2'] = np.mean(score_slope['test_r2'])
                df_tmp_score.loc[0,'slope_mae'] = -np.mean(score_slope['test_neg_mean_absolute_error'])
                df_tmp_score.loc[0,'slope_r2_std'] = np.std(score_slope['test_r2'])
                df_tmp_score.loc[0,'slope_mae_std'] = np.std(score_slope['test_neg_mean_absolute_error'])
            
            if np.sum(bb) == 0:
                df_tmp_score.loc[0,'intercept_r2'] = 1
                df_tmp_score.loc[0,'intercept_mae'] = 0
                df_tmp_score.loc[0,'intercept_r2_std'] = 0
                df_tmp_score.loc[0,'intercept_mae_std'] = 0
            else: 
                score_intercept = cross_validate(Krige(**param_dict),
                                                  X, bb, 
                                                  scoring=['r2', 'neg_mean_absolute_error'], 
                                                  cv=cv, groups=groups, n_jobs=-1, verbose=0)
                df_tmp_score.loc[0,'intercept_r2'] = np.mean(score_intercept['test_r2'])
                df_tmp_score.loc[0,'intercept_mae'] = -np.mean(score_intercept['test_neg_mean_absolute_error'])
                df_tmp_score.loc[0,'intercept_r2_std'] = np.std(score_intercept['test_r2'])
                df_tmp_score.loc[0,'intercept_mae_std'] = np.std(score_intercept['test_neg_mean_absolute_error'])
            # Store to DataFrame
            df_tmp_score.loc[0,'feature'] = feature
            df_CV_score = df_CV_score.append(df_tmp_score, ignore_index=True)


#%% Save results to pickle
path_dict = '../../09-Results/Stage-01/LocalLinearFitwBounds-UKriging-GroupKfold-Scores.pkl'
save_obj(path_dict, df_CV_score)
path_dict = '../../09-Results/Stage-01/LocalLinearFitwBounds-UKriging-GroupKfold-Results.pkl'
save_obj(path_dict, df_CV_pred)

print('Done')

















