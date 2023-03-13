# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 12:03:48 2022

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
df_CV_pred = pd.DataFrame([])
for feature in ['qc', 'fs', 'u2']:
    print('Data: ', feature)
    # Loop over units
    for unit in unitlist:
        print('\tUnit: ', unit)
        loc = df_lreg.loc[(df_lreg['feature']==feature) & (df_lreg['unit_geo']==unit), ['ID']].values.astype(int).flatten()
        # X = df_lreg.loc[(df_lreg['feature']==feature) & (df_lreg['unit_geo']==unit), ['x','y']].values.astype(float)
        # aa = df_lreg.loc[(df_lreg['feature']==feature) & (df_lreg['unit_geo']==unit), 'slope']
        # aa = np.array([x for x in aa]).flatten()
        # bb = df_lreg.loc[(df_lreg['feature']==feature) & (df_lreg['unit_geo']==unit), 'intercept']
        # bb = np.array([x for x in bb]).flatten()
        
        for iloc in loc:
            df_tmp_pred = pd.DataFrame([])
            X = df_lreg.loc[(df_lreg['feature']==feature) & (df_lreg['unit_geo']==unit) & (df_lreg['ID']!=iloc), ['x','y']].values.astype(float)
            aa = df_lreg.loc[(df_lreg['feature']==feature) & (df_lreg['unit_geo']==unit) & (df_lreg['ID']!=iloc), 'slope']
            aa = np.array([x for x in aa]).flatten()
            bb = df_lreg.loc[(df_lreg['feature']==feature) & (df_lreg['unit_geo']==unit) & (df_lreg['ID']!=iloc), 'intercept']
            bb = np.array([x for x in bb]).flatten()
            
            df_tmp_pred.loc[0,'ID'] = iloc
            df_tmp_pred.loc[0,'unit'] = unit
            df_tmp_pred.loc[0,'feature'] = feature
            
            X_loc = df_lreg.loc[(df_lreg['feature']==feature) & (df_lreg['unit_geo']==unit) & (df_lreg['ID']==iloc), ['x','y']].values.astype(float)
            
            aa_loc = df_lreg.loc[(df_lreg['feature']==feature) & (df_lreg['unit_geo']==unit) & (df_lreg['ID']==iloc), 'slope'].values
            bb_loc = df_lreg.loc[(df_lreg['feature']==feature) & (df_lreg['unit_geo']==unit) & (df_lreg['ID']==iloc), 'intercept'].values
            df_tmp_pred['aa_loc'] = aa_loc
            df_tmp_pred['bb_loc'] = bb_loc
            
            # if np.shape(X)[0]>1:
            if np.abs(np.sum(np.diff(aa)))<1e-6:
                df_tmp_pred['aa_loo_loc'] = aa[0]
                df_tmp_pred['aa_loo_loc_var'] = 0
            else:
                UK_slope = UniversalKriging(X[:,0], X[:,1], aa, **param_dict_K)
                aa_loo_loc, aa_loo_loc_var = UK_slope.execute('points', X_loc[:,0], X_loc[:,1])
                df_tmp_pred['aa_loo_loc'] = aa_loo_loc
                df_tmp_pred['aa_loo_loc_var'] = aa_loo_loc_var
                
            if np.sum(bb) == 0:
                df_tmp_pred['bb_loo_loc'] = 0
                df_tmp_pred['bb_loo_loc_var'] = 0
            else:
                UK_intercept = UniversalKriging(X[:,0], X[:,1], bb, **param_dict_K)
                bb_loo_loc, bb_loo_loc_var = UK_intercept.execute('points', X_loc[:,0], X_loc[:,1])
                df_tmp_pred['bb_loo_loc'] = bb_loo_loc
                df_tmp_pred['bb_loo_loc_var'] = bb_loo_loc_var
            

            # df_tmp_pred.loc[0,'loo_slope_uk'] = UK_slope
            # df_tmp_pred.loc[0,'loo_intercept_uk'] = UK_intercept            
            # aa_loo_loc, aa_loo_loc_var = UK_slope.execute('points', X_loc[:,0], X_loc[:,1])
            # bb_loo_loc, bb_loo_loc_var = UK_intercept.execute('points', X_loc[:,0], X_loc[:,1])

            


            df_CV_pred = df_CV_pred.append(df_tmp_pred, ignore_index=True)        

#%% Save results to pickle
path_dict = '../../09-Results/Stage-01/LocalLinearFitwBounds-UKriging-LOO_Estimator.pkl'
save_obj(path_dict, df_CV_pred)

print('Done')

















