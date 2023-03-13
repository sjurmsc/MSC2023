# -*- coding: utf-8 -*-
"""
Created on Tue May  4 23:46:48 2021

@author: GuS
"""

#%% Import libraries
import numpy as np
import pandas as pd
import pickle
import time

from pykrige.rk import Krige
from pykrige.uk3d import UniversalKriging3D
from pykrige.ok3d import OrdinaryKriging3D

from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.model_selection import cross_validate, cross_val_predict

#%% Define functions
def load_obj(path_obj):
    with open(path_obj, 'rb') as f:
        return pickle.load(f)
    
def save_obj(path_obj, obj):
    with open(path_obj, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
       
#%% Load database
path_database = '../../09-Results/Stage-01/Database_upscaled.csv'
df = pd.read_csv(path_database)

# list of units
# unitlist = df['unit_geo'].dropna().unique()
unitlist = df['unit'].dropna().unique()

#%% Group Kfold and leaveOneGroupOut cross-validation Kriging
# Kriging parameters
param_dict = {
    'method': 'universal3d',
    'variogram_model': 'spherical',
    'nlags': 32,
    'anisotropy_scaling': (1,5000),
    'weight': True,
    'drift_terms': 'regional_linear',
    'pseudo_inv': False,
    }
param_dict_K = {
    'variogram_model': 'spherical',
    'nlags': 32,
    'anisotropy_scaling_y': 1,
    'anisotropy_scaling_z': 5000,
    'weight': True,
    'drift_terms': 'regional_linear',
    'enable_plotting': False,
    'pseudo_inv': False,
    'verbose': 0
    }

unitlist = np.sort(unitlist)

# Loop over features
df_CV_score = pd.DataFrame([])
df_CV_pred = pd.DataFrame([])
for feature in ['qc']:#, 'fs', 'u2']:
    print('Data: ', feature)
    # Loop over units
    for unit in unitlist[:-1]:
    # for unit in ['GGM23']:
    # for unit in ['GGM24', 'GGM31B', 'GGM51']:
        # param_dict = {
        #     'method': 'universal3d',
        #     'variogram_model': 'spherical',
        #     'nlags': 32,
        #     'anisotropy_scaling': (1,1),
        #     'weight': True,
        #     'drift_terms': 'regional_linear',
        #     'pseudo_inv': False,
        #     }
        # param_dict_K = {
        #     'variogram_model': 'gaussian',
        #     'nlags': 32,
        #     'anisotropy_scaling_y': 1,
        #     'anisotropy_scaling_z': 500,
        #     'weight': True,
        #     'drift_terms': None,
        #     'enable_plotting': False,
        #     'pseudo_inv': True,
        #     'exact_values': True,
        #     'verbose': 0
        #     }
        param_dict_K = {
            'variogram_model': 'spherical',
            'nlags': 40,
            'anisotropy_scaling_y': 1,
            'anisotropy_scaling_z': 500,
            'weight': True,
            'enable_plotting': False,
            'pseudo_inv': True,
            'exact_values': True,
            'verbose': 0
            }
        print('\tUnit:', unit)
        df_tmp_score = pd.DataFrame([])
        df_tmp_pred = pd.DataFrame([])
        # feature_list=['x','y','unit_geo','z_bsf', feature, 'ID']
        # XX = df.loc[df['unit_geo']==unit, feature_list].dropna(subset=[feature])
        feature_list=['x','y','unit','z_bsf', feature, 'ID']
        # XX = df.loc[df['unit_geo']==unit, feature_list].dropna(subset=[feature])
        XX = df.loc[df['unit']==unit, feature_list].dropna(subset=[feature])
        groups = XX['ID'].values.flatten()
        # X = XX.drop(columns=[feature, 'unit_geo', 'ID']).values
        # y = df.loc[df['unit_geo']==unit, feature].dropna().values
        X = XX.drop(columns=[feature, 'unit', 'ID']).values
        # y = df.loc[df['unit_geo']==unit, feature].dropna().values
        y = df.loc[df['unit']==unit, feature].dropna().values
        # if (unit == 'GGM51'):
        #     param_dict = {
        #         'method': 'ordinary3d',
        #         'variogram_model': 'spherical',
        #         'nlags': 32,
        #         'anisotropy_scaling': (10,5000),
        #         'weight': True,
        #         'pseudo_inv': False,
        #         }
        #     param_dict_K = {
        #         'variogram_model': 'spherical',
        #         'nlags': 32,
        #         'anisotropy_scaling_y': 10,
        #         'anisotropy_scaling_z': 5000,
        #         'weight': True,
        #         'enable_plotting': False,
        #         'pseudo_inv': False,
        #         'verbose': 0
        #         }
        # if (unit == 'GGM24') | (unit == 'GGM31B') | (unit == 'GGM53') | (unit == 'GGM54'):
        #     param_dict = {
        #         'method': 'ordinary3d',
        #         'variogram_model': 'spherical',
        #         'nlags': 32,
        #         'anisotropy_scaling': (1,5000),
        #         'weight': True,
        #         'pseudo_inv': False,
        #         }
        #     param_dict_K = {
        #         'variogram_model': 'spherical',
        #         'nlags': 32,
        #         'anisotropy_scaling_y': 1,
        #         'anisotropy_scaling_z': 5000,
        #         'weight': True,
        #         'enable_plotting': False,
        #         'pseudo_inv': False,
        #         'verbose': 0
        #         }
        
        start = time.time()
        # 3D Kriging full dataset
        print('\t\t3D Kriging - full dataset')
        # if (unit == 'GGM24') | (unit == 'GGM31B') | (unit == 'GGM51') | (unit == 'GGM53') | (unit == 'GGM54'):
        #     UK3D = OrdinaryKriging3D(X[:,0], X[:,1], X[:,2], y, **param_dict_K)
        # else:
        #     UK3D = UniversalKriging3D(X[:,0], X[:,1], X[:,2], y, **param_dict_K)
        # UK3D = UniversalKriging3D(X[:,0], X[:,1], X[:,2], y, **param_dict_K)
        UK3D = OrdinaryKriging3D(X[:,0], X[:,1], X[:,2], y, **param_dict_K)
        
        UK3D.display_variogram_model()
        
        df_tmp_score['RFreg_obj'] = [UK3D]
        df_tmp_score['feature'] = [feature]
        df_tmp_score['unit'] = unit
        
        # LeaveOneGroupOut Cross validation prediction 3DKriging
        # print('\t\t3D Kriging - LeaveOneOut prediction')
        # cv = LeaveOneGroupOut()
        # df_tmp_pred = XX.copy()
        # df_tmp_pred['y_true'] = y
        # df_tmp_pred['y_pred_loo'] = cross_val_predict(Krige(**param_dict),
        #                                               X, y, cv=cv, groups=groups,
        #                                               n_jobs=-1, verbose=0)
        # df_tmp_pred['feature'] = feature
        # df_CV_pred = df_CV_pred.append(df_tmp_pred, ignore_index=True)
  
        # # GroupKfold Cross validation 3D Kriging scoring  
        # print('\t\t3D Kriging - Kfold and LeaveOneOut Scoring')
        # n_splits = 6
        # # cv = GroupKFold(n_splits=n_splits)
        # cv = LeaveOneGroupOut()
        # score = cross_validate(Krige(**param_dict),
        #                         X, y, scoring=['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error'],
        #                         cv=cv, groups=groups, n_jobs=-1, verbose=0)
        # df_tmp_score['r2_kfold'] = np.mean(score['test_r2'])
        # df_tmp_score['mae_kfold'] = -np.mean(score['test_neg_mean_absolute_error'])
        # df_tmp_score['mse_kfold'] = -np.mean(score['test_neg_mean_squared_error'])
        # df_tmp_score['r2_kfold_std'] = np.std(score['test_r2'])
        # df_tmp_score['mae_kfold_std'] = np.std(score['test_neg_mean_absolute_error'])
        # df_tmp_score['mse_kfold_std'] = np.std(score['test_neg_mean_squared_error'])
            
        df_CV_score = df_CV_score.append(df_tmp_score, ignore_index=True)

        end = time.time()
        print('Elapsed time: ', end - start)

        
#%% Save results to pickle
# path_dict = '../../09-Results/Stage-01/UKriging3D-Kfold-Scores.pkl'
# save_obj(path_dict, df_CV_score)
# path_dict = '../../09-Results/Stage-01/UKriging3D-Kfold-Results.pkl'
# save_obj(path_dict, df_CV_pred)
        
        
        
        
        
        
        
        
        