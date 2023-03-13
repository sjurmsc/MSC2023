# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 15:57:16 2021

@author: GuS
"""

#%% import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pykrige.uk import UniversalKriging


#%% Load hor picks
# path_picks = "P:/2019/07/20190798/Background/SAND_Geophysics/2021-06-16/01_Picks/SAND_R20_Z1edit.dat"
path_picks = "P:/2019/07/20190798/Background/SAND_Geophysics/2021-06-16/01_Picks/SAND_R23_Z1edit.dat"
df_picks = pd.read_csv(path_picks, sep='\s+', header=None, names=['line', 'u', 'unit', 'x', 'y', 'z'])

# Interpolation kriging
param_dict_K = {
    'variogram_model': 'linear',
    'nlags': 32,
    'weight': True,
    'drift_terms': 'regional_linear',
    'enable_plotting': False,
    'pseudo_inv': False,
    'verbose': 0
    }

x = df_picks['x'].values
y = df_picks['y'].values
z = -df_picks['z'].values

UK2D = UniversalKriging(x, y, z, **param_dict_K)

