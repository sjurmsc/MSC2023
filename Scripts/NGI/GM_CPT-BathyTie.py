# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 14:34:12 2021

Function to get CPT 0 m [MD] from bathy depth and estimate TotalDepth from CPT data (LAS files)

@author: GuS
"""

#%% Import libraries
# import numpy as np
import os
import netCDF4 as nc
import numpy as np
import pandas as pd
import scipy.spatial as spatial

import GM_LoadData as GML

#%% load CPT locations
# CPT locations
filepath = 'P:/2019/07/20190798/Calculations/GroundModel/01-Data/GenerateBHlogs.xlsx'
df_CPT_loc = pd.DataFrame([])
df_tmp = pd.read_excel(filepath, sheet_name='Data', header=None, usecols=[7,8,9,10], skiprows=14, names=['borehole','x','y','WD'])
df_CPT_loc = df_CPT_loc.append(df_tmp)
df_tmp = pd.read_excel(filepath, sheet_name='Data', header=None, usecols=[13, 14, 15, 16], skiprows=14, names=['borehole','x','y','WD'])
df_tmp.dropna(inplace=True)
df_CPT_loc = df_CPT_loc.append(df_tmp)
df_tmp = pd.read_excel(filepath, sheet_name='Data', header=None, usecols=[19, 20, 21, 22], skiprows=14, names=['borehole','x','y','WD'])
df_tmp.dropna(inplace=True)
df_CPT_loc = df_CPT_loc.append(df_tmp)
df_tmp = pd.read_excel(filepath, sheet_name='Data', header=None, usecols=[25, 26, 27, 28], skiprows=14, names=['borehole','x','y','WD'])
df_tmp.dropna(inplace=True)
df_CPT_loc = df_CPT_loc.append(df_tmp)
df_tmp = pd.read_excel(filepath, sheet_name='Data', header=None, usecols=[31, 32, 33, 34], skiprows=14, names=['borehole','x','y','WD'])
df_tmp.dropna(inplace=True)
df_CPT_loc = df_CPT_loc.append(df_tmp)
# # Rename borehole
# df_CPT_loc['borehole'] = df_CPT_loc['borehole'].str.slice(stop=6)
# Indexing on borehole
df_CPT_loc.set_index('borehole', inplace=True)

#%% Load CPT data (LAS files)
# directory = 'P:/2019/07/20190798/Calculations/LasFiles/lasfiles_phase1_phase2/combined'
directory = 'P:/2019/07/20190798/Calculations/LasFiles/lasfiles_phase1_phase2'
df_CPT_data = GML.load_cpts_qc_LASIO(directory, cptloc=df_CPT_loc)
# Indexing on borehole
df_CPT_data.set_index('borehole', inplace=True)

del filepath, directory

#%% Load Seafloor
filepath = "P:/2019/07/20190798/Calculations/Bathymetry/MMT_RVO_TNW_EM2040D_5m_alldata.grd"
fh = nc.Dataset(filepath, mode='r')
x = fh.variables['x'][:].filled()
y = fh.variables['y'][:].filled()
z = fh.variables['z'][:].filled()
fh.close()

del filepath, fh

#%% For each CPT find closest point in bathy and assigned depth
#%% Format data for quadtree decomposition
xx, yy = np.meshgrid(np.float32(x), np.float32(y))
xy = np.c_[xx.ravel(), yy.ravel()]
zz = np.c_[z.ravel()]

#%% quadtree decomposition of the bathy
tree = spatial.KDTree(xy)
# tree = spatial.KDTree(hor[['x', 'y','z']])

#%% loop over all CPT locations find closest location
for cpt, row in df_CPT_loc.iterrows():
    print('cpt: ' + str(cpt))
    distance, index = tree.query(row[['x','y']])
    df_CPT_loc.at[cpt, 'WD'] = -zz[index]    
    
    
#%% Estimate Total depth for each CPT
df_CPT_loc['TD'] = np.nan
for cpt, row in df_CPT_loc.iterrows():
    if cpt in df_CPT_data.index:
        print('cpt: ' + str(cpt))
        df_CPT_loc.loc[cpt,'TD'] = df_CPT_loc.loc[cpt, 'WD'] + int(np.max(df_CPT_data.loc[cpt, ['MD']])[0]+1)
        df_CPT_loc.loc[cpt,'TotalDepth'] = int(np.max(df_CPT_data.loc[cpt, ['MD']])[0]+1)
    else:
        df_CPT_loc.loc[cpt,'TD'] = df_CPT_loc.loc[cpt, 'WD'] + 100
        df_CPT_loc.loc[cpt,'TotalDepth'] = 100
del cpt
    
#%% Export to CSV file
# df_CPT_loc.to_csv("../../01-Data/MergeWellTrack.csv", sep='\t')  
df_CPT_loc[['x','y','WD', 'TD', 'TotalDepth']].to_csv("../../01-Data/AllWellTrack.csv", sep='\t')  

#%% Export CPT data location only to csv
directory = 'P:/2019/07/20190798/Calculations/LasFiles/lasfiles_phase1_phase2'
borehole_list = []
for filename in os.listdir(directory):
        if filename.endswith(".las" or ".LAS"):
            borehole_list.append(os.path.splitext(filename)[0])

df_sel = df_CPT_loc.loc[borehole_list]
df_sel[['x','y','WD','TD','TotalDepth']].to_csv("../../01-Data/CPTWellTrack.csv", sep='\t')              



