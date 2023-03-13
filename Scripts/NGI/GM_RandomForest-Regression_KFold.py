# -*- coding: utf-8 -*-
"""
Created on Tue May  4 10:43:30 2021

@author: GuS
"""

#%% Import libraries
import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.model_selection import cross_validate, cross_val_predict

#%% Define functions
def load_obj(path_obj):
    with open(path_obj, 'rb') as f:
        return pickle.load(f)
    
def save_obj(path_obj, obj):
    with open(path_obj, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def unit2num(df,col_name):
    u=[]
    for st in df['unit']:
        if isinstance(st,float):
            u.append(999)
        else:
            if 'GBB' == st[:3]:
               if 'A' in st[-1]: 
                   u.append(float(st[3:-1])+300)
               elif 'B' in st[-1]:
                   u.append(float(st[3:-1])+400)               
            else:
                if 'A' in st[-1]:    
                    u.append(float(st[3:-1])+100)
                elif 'B' in st[-1]:
                    u.append(float(st[3:-1])+200)
                else:
                    u.append(float(st[3:]))
    df[col_name+'_no']=np.array(u) 
    return(df)

#%% Load database
# path_database = '../../09-Results/Stage-01/Database_mean.csv'
path_database = '../../09-Results/Stage-01/Database.csv'
df = pd.read_csv(path_database)

#Convert unit string to unit number
df = unit2num(df, 'unit')

#%% Group Kfold and leaveOneGroupOut cross-validation Random forest regression
# Random Forest hyperparameters
param_dict = {
    'max_depth': 20,
    'n_estimators': 20,
    'min_samples_leaf': 1,
    'min_samples_split': 4,
    'bootstrap': True,
    'criterion': 'mse'
    }

# Loop over features
df_CV_score = pd.DataFrame([])
df_CV_pred = pd.DataFrame([])
for feature in ['qc', 'fs', 'u2']:
    print('Data: ', feature)
    df_tmp_score = pd.DataFrame([])
    df_tmp_pred = pd.DataFrame([]) 
    
#    feature_list=['z_bsl','x','y','unit','unit_no','z_bsf','envelop','energy', feature, 'ID']
    df.dropna(inplace=True)
    feature_list=['z_bsl','x','y','unit','unit_no','z_bsf','Qp','Vp','envelop','energy', feature, 'ID']
    XX = df.loc[:,feature_list].dropna(subset=[feature])
    groups = XX['ID'].values.flatten()
    X = XX.drop(columns=[feature, 'unit', 'ID'])
    y = df.loc[:,feature].dropna()
    
    # Random forest full dataset
    print('\tRandom forest regression - full dataset')
    RFreg = RandomForestRegressor(**param_dict)
    RFreg.fit(X, y)
    df_tmp_score['RFreg_obj'] = [RFreg]
    df_tmp_score['feature'] = feature
    
    # LeaveOneGroupOut Cross validation prediction RFreg
    print('\tRandom forest regression - LeaveOneOut prediction')
    cv = LeaveOneGroupOut()
    df_tmp_pred = XX.copy()
    df_tmp_pred['y_true'] = y
    df_tmp_pred['y_pred_loo'] = cross_val_predict(RandomForestRegressor(**param_dict), 
                                              X, y, cv=cv, groups=groups, 
                                              n_jobs=-1, verbose=0)
    df_tmp_pred['feature'] = feature
    df_CV_pred = df_CV_pred.append(df_tmp_pred, ignore_index=True)
        
    # GroupKfold Cross validation RFreg scoring  
    print('\tRandom forest regression - Kfold and LeaveOneOut Scoring')
    n_splits = 6
    # cv = GroupKFold(n_splits=n_splits)
    cv = LeaveOneGroupOut()
    score = cross_validate(RandomForestRegressor(**param_dict),
                            X, y, scoring=['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error'],
                            cv=cv, groups=groups, n_jobs=-1, verbose=0)
    df_tmp_score['r2_kfold'] = np.mean(score['test_r2'])
    df_tmp_score['mae_kfold'] = -np.mean(score['test_neg_mean_absolute_error'])
    df_tmp_score['mse_kfold'] = -np.mean(score['test_neg_mean_squared_error'])
    df_tmp_score['r2_kfold_std'] = np.std(score['test_r2'])
    df_tmp_score['mae_kfold_std'] = np.std(score['test_neg_mean_absolute_error'])
    df_tmp_score['mse_kfold_std'] = np.std(score['test_neg_mean_squared_error'])
        
    # # LeaveOneGroupOut Cross validation RFreg scoring
    # cv = LeaveOneGroupOut()
    # score = cross_validate(RandomForestRegressor(**param_dict),
    #                         X, y, scoring=['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error'],
    #                         cv=cv, groups=groups, n_jobs=-1, verbose=0)
    # df_tmp_score['r2_loo'] = np.mean(score['test_r2'])
    # df_tmp_score['mae_loo'] = -np.mean(score['test_neg_mean_absolute_error'])
    # df_tmp_score['mse_loo'] = -np.mean(score['test_neg_mean_squared_error'])
    # df_tmp_score['r2_loo_std'] = np.std(score['test_r2'])
    # df_tmp_score['mae_loo_std'] = np.std(score['test_neg_mean_absolute_error'])
    # df_tmp_score['mse_loo_std'] = np.std(score['test_neg_mean_squared_error'])
    
    df_CV_score = df_CV_score.append(df_tmp_score, ignore_index=True)
    
#%% Save results to pickle
path_dict = '../../09-Results/Stage-01/RandomForest-Kfold-Scores.pkl'
save_obj(path_dict, df_CV_score)
path_dict = '../../09-Results/Stage-01/RandomForest-Kfold-Results.pkl'
save_obj(path_dict, df_CV_pred)











