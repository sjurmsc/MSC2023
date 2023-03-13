# -*- coding: utf-8 -*-
"""
Created on Wed May  5 09:02:56 2021

@author: GuS
"""

#%% Import libraries
import pandas as pd
import numpy as np

#%% Define functions


#%% Load database
# path_database = '../../09-Results/Stage-01/Database.pkl'
# df = pd.read_pickle(path_database)
path_database = '../../09-Results/Stage-01/Database.csv'
df = pd.read_csv(path_database)
#%% Main
df_mean = pd.DataFrame([])
for loc in df['ID'].unique():
    print('Loc: ', loc)
    df_loc = df.loc[df['ID']==loc,:]
    df_unit = df_loc.drop_duplicates(subset=['z_bsl'])
    df_loc_mean = df_loc.groupby('z_bsl', as_index=False).mean()
    df_m = df_loc_mean.merge(df_unit[['z_bsl', 'unit', 'unit_geo', 'envelop', 'energy', 'cumulative_envelop', 'cumulative_energy']], how='left', on='z_bsl')
    df_mean = df_mean.append(df_m, ignore_index=True)
        

#%% Export database to csv
path_datacsv = '../../09-Results/Stage-01/Database_mean.csv'
print('Writing to csv ...')
df_mean.to_csv(path_datacsv, index=False)
print('csv file written: ', path_datacsv)

path_datapkl = '../../09-Results/Stage-01/Database_mean.pkl'
print('Writing to pickle ...')
df_mean.to_pickle(path_datapkl)
print('csv file written: ', path_datapkl)