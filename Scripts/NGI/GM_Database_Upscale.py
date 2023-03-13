# -*- coding: utf-8 -*-
"""
Created on Wed May  5 09:02:56 2021

@author: GuS
"""

#%% Import libraries
import pandas as pd
import numpy as np


#%% Define functions
def moving_average(a, n=4):
    if (n % 2) != 0:
        n=n+1
    ma=np.copy(a)
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ma[int(n/2-1):int(-n/2)]=ret[n - 1:] / n
    return ma

def upscale_database(df, upscale=0.5, n=50):
    df_qcU = pd.DataFrame([])
    df_fsU = pd.DataFrame([])
    df_u2U = pd.DataFrame([])
    for loc in df['ID'].unique():
        print(loc)
        df_loc = df[df['ID']==loc]
        df_loc = df_loc.sort_values(by=['z_bsf'])
        x = df_loc['x'].mean()
        y = df_loc['y'].mean()
        
        # qc 
        df_data = df_loc.dropna(subset=['qc'])
        z = df_data['z_bsf'].values        
        unit = df_data['unit'].values
        unit_geo = df_data['unit_geo'].values
        inx = np.linspace(0,len(z)-1,len(z)).astype(int)        
        z_U = np.arange(np.round(z.min()), np.round(z.max()), upscale)                 
        qc = df_data['qc'].values            
        qcf = np.interp(z, moving_average(z, n), moving_average(qc, n))         
        qc_U = np.interp(z_U, z, qcf)        
        inx_U = np.interp(z_U, z, inx).astype(int)
        unit_U = unit[inx_U]
        unit_geo_U  = unit_geo[inx_U]
        df_qc = pd.DataFrame(data={'ID': loc, 'x': x, 'y': y, 'z_bsf': z_U, 'qc': qc_U, 'unit':unit_U, 'unit_geo':unit_geo_U})
        
        # fs
        df_data = df_loc.dropna(subset=['fs'])
        z = df_data['z_bsf'].values        
        unit = df_data['unit'].values
        unit_geo = df_data['unit_geo'].values
        inx = np.linspace(0,len(z)-1,len(z)).astype(int)        
        z_U = np.arange(np.round(z.min()), np.round(z.max()), upscale)            
        fs = df_data['fs'].values            
        fsf = np.interp(z, moving_average(z, n), moving_average(fs, n))         
        fs_U = np.interp(z_U, z, fsf)        
        inx_U = np.interp(z_U, z, inx).astype(int)
        unit_U = unit[inx_U]
        unit_geo_U  = unit_geo[inx_U]
        df_fs = pd.DataFrame(data={'ID': loc, 'z_bsf': z_U, 'fs': fs_U, 'unit':unit_U, 'unit_geo':unit_geo_U})
        
        # u2
        df_data = df_loc.dropna(subset=['u2'])
        z = df_data['z_bsf'].values        
        unit = df_data['unit'].values
        unit_geo = df_data['unit_geo'].values
        inx = np.linspace(0,len(z)-1,len(z)).astype(int)        
        z_U = np.arange(np.round(z.min()), np.round(z.max()), upscale)            
        u2 = df_data['u2'].values            
        u2f = np.interp(z, moving_average(z, n), moving_average(u2, n))         
        u2_U = np.interp(z_U, z, u2f)        
        inx_U = np.interp(z_U, z, inx).astype(int)
        unit_U = unit[inx_U]
        unit_geo_U  = unit_geo[inx_U]
        df_u2 = pd.DataFrame(data={'ID': loc, 'z_bsf': z_U, 'u2': u2_U, 'unit':unit_U, 'unit_geo':unit_geo_U})
        
       
        df_qcU = df_qcU.append(df_qc, ignore_index=True)
        df_fsU = df_fsU.append(df_fs, ignore_index=True)
        df_u2U = df_u2U.append(df_u2, ignore_index=True)
        
    df_tmp = df_qcU.merge(df_fsU, how='left', on=['ID', 'z_bsf'], suffixes=(None, '_y'))
    dfU = df_tmp.merge(df_u2U, how='left', on=['ID', 'z_bsf'], suffixes=(None, '_y'))
    dfU = dfU.drop(columns=['unit_y', 'unit_geo_y'])
    
    return dfU



#%% Load database
# path_database = '../../09-Results/Stage-01/Database_mean.pkl'
# df = pd.read_pickle(path_database)
path_database = '../../09-Results/Stage-01/Database_mean.csv'
df = pd.read_csv(path_database)
#%% Main
df_upscaled = upscale_database(df, upscale=0.25, n=50)
        
    
#%% Export database to csv
path_datacsv = '../../09-Results/Stage-01/Database_upscaled.csv'
print('Writing to csv ...')
df_upscaled.to_csv(path_datacsv, index=False)
print('csv file written: ', path_datacsv)

path_datapkl = '../../09-Results/Stage-01/Database_upscaled.pkl'
print('Writing to pickle ...')
df_upscaled.to_pickle(path_datapkl)
print('csv file written: ', path_datapkl)