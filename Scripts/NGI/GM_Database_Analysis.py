# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 09:52:01 2021

Load database and perform statistical analysis of data

@author: GuS
"""

#%% import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#%% Load database
path_database = '../../09-Results/Stage-01/Database.csv'
df = pd.read_csv(path_database)

#%% Print data summary statistic and write to csv
df_desc = df.iloc[:,:-1].describe()
print(df_desc.transpose())
path_dfdesc = '../../09-Results/Stage-01/Database_Stats.csv'
df_desc.to_csv(path_dfdesc, index=False)


#%% Plot data histogram
df_h = df.iloc[:,2:-1].copy()
df_h.hist(bins=50, figsize=(18,18))
path_histplot = '../../09-Results/Stage-01/Database_hist.png'
plt.tight_layout()
plt.savefig(path_histplot, dpi=200) 

#%% Plot matrix of scatter plots of all features versus all features
# The diagonal shows the histogram of the feature
df_h = df.iloc[:,2:-1].copy()
df_h.dropna(inplace=True)
pd.plotting.scatter_matrix(df_h.iloc[:,2:-1], s=1, c=df_h['ID'], alpha=0.5, figsize=(18,18), diagonal='hist')
plt.tight_layout()
path_scatplot = '../../09-Results/Stage-01/Database_ScatterXplot.png'
plt.savefig(path_scatplot, dpi=200) 


#%% Plot matrix of scatter plots of selection of features
# The diagonal shows the histogram of the feature
df_h = df[['ID', 'z_bsf', 'unit', 'unit_geo', 'qc', 'fs', 'u2', 'qt', 'Qt', 'Bq', 'Fr', 'ICN', 'Qp', 'Vp',
           'amp_tr_1', 'envelop', 'energy', 'cumulative_envelop', 'cumulative_energy']]
df_h.dropna(inplace=True)
pd.plotting.scatter_matrix(df_h.iloc[:,1:], s=1, c=df_h['ID'], alpha=0.5, figsize=(20,20), diagonal='hist')
plt.tight_layout()
path_scatplot = '../../09-Results/Stage-01/Database_ScatterXplot_reduced.png'
plt.savefig(path_scatplot, dpi=200) 


#%% Plot correlation matrix
df_corr = df[['z_bsf', 'unit', 'unit_geo', 'qc', 'fs', 'u2', 'qt', 'Qt', 'Bq', 'Fr', 'ICN', 'Qp', 'Vp', 'amp_tr_1', 'envelop', 'energy', 'cumulative_envelop', 'cumulative_energy']].corr()
plt.figure(figsize=(12,10))
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(df_corr, dtype=bool), k=1)
sns.heatmap(df_corr, mask=mask, annot=True, fmt='0.1f', vmin=-1, vmax=1, center=0)
plt.tight_layout()
plt.show
path_corrplot = '../../09-Results/Stage-01/Database_CorrelationMatrix.png'
plt.savefig(path_corrplot, dpi=200) 
