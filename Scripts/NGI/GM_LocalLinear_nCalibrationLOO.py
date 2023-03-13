# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 13:36:21 2022

@author: GuS
"""


#%% Import libraries
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker

import GM_Toolbox as GMT 

#%% Define functions
def load_obj(path_obj):
    with open(path_obj, 'rb') as f:
        return pickle.load(f)
    
def plot_data_fit_krig(ax, X, y, X_pred, y_pred, Xy_pred, feature):
    model_dist, mae, accuracy, mu, std, mape = GMT.evaluate_modeldist_norm(y, Xy_pred)
    ax.plot(y_pred, X_pred, 'k', linestyle='--', linewidth=0.75)
    X_std = np.array([X_pred.min(), X_pred.max()])
    y_std = np.array([y_pred.min(), y_pred.max()]).flatten()
    ax.fill_betweenx(X_std, y_std*(1-std), y_std*(1+std), alpha=0.2, color='k', edgecolor='k')
    return ax

def show_values(ax, pc, fmt="%.1f", **kw):
    pc.update_scalarmappable()
    # ax = pc.get_axes()
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array().data):
        # print(value)
        if ~np.isnan(value):
            x, y = p.vertices[:-2, :].mean(0)
            if value>=0.2:
                color = (0.0, 0.0, 0.0)
            else:
                color = (1.0, 1.0, 1.0)
            ax.text(x-0.18, y+0.18, fmt % value, ha="center", va="center", color=color, fontsize=9, **kw)


#%% Load database
path_database = '../../09-Results/Stage-01/Database.csv'
df = pd.read_csv(path_database)
# Limit depth
z_min, z_max = -1, 80
# z_min, z_max = df['z_bsf'].min(), df['z_bsf'].max()
df = df[(df['z_bsf']>=z_min) & (df['z_bsf']<=z_max)]

#%% Load mean database
path_mean = '../../09-Results/Stage-01/Database_mean.csv'
df_mean = pd.read_csv(path_mean)
df_mean = df_mean[(df_mean['z_bsf']>=z_min) & (df_mean['z_bsf']<=z_max)]


#%% Load local linear fit KFold results and Kriged results
path_dict = '../../09-Results/Stage-01/LocalLinearFit-Kfold-Results_bounds.pkl'
df_lreg = load_obj(path_dict)
path_dict = '../../09-Results/Stage-01/LocalLinearFitwBounds-UKriging-GroupKfold-Results.pkl'
df_lregKri = load_obj(path_dict)
path_dict = '../../09-Results/Stage-01/LocalLinearFitwBounds-UKriging-GroupKfold-Scores.pkl'
df_lregKriM = load_obj(path_dict)
path_dict = '../../09-Results/Stage-01/LocalLinearFitwBounds-UKriging-LOO_Estimator.pkl'
df_loo = load_obj(path_dict)

#%% Main loop over CPT location (ID)
# List of locations and units
loclist = df['ID'].sort_values().unique()
unitlist = df['unit_geo'].dropna().unique()

# Loop over locations
# df_true = pd.DataFrame([])
# df_pred = pd.DataFrame([])
df_truepred = pd.DataFrame([])
for loc in loclist:
    iloc = int(loc)
    print('Loc:', iloc)
    # get all CPT at this location
    df_loc = df_mean.loc[df_mean['ID']==loc,:]
    df_loo_loc = df_loo.loc[df_loo['ID']==loc,:]
    # Plot linear fit
    # for ii, feature in zip([0],['qc']):
    for ii, feature in zip([0, 1, 2],['qc', 'fs', 'u2']):
        print(feature)
        # Loop over units
        for unit in df_loo_loc['unit'].dropna().unique():
            print(unit)
            df_tmp = pd.DataFrame()
            df_tmp_pred = pd.DataFrame()
            df_data = df_loc.loc[df_loc['unit']==unit,['z_bsf', feature]].dropna()
            if len(df_data)>1:
                Xy = df_data[['z_bsf', feature]]#.drop_duplicates(subset=['z_bsf'])
                X = Xy['z_bsf'].values.reshape(-1,1)
                y = Xy[feature].values.reshape(-1,1)
                
                ##
                aa_loo = df_loo.loc[(df_loo['feature']==feature) & (df_loo['unit']==unit) & (df_loo['ID']==iloc), 'aa_loo_loc'].values
                if aa_loo.size == 0:
                    break
                bb_loo = df_loo.loc[(df_loo['feature']==feature) & (df_loo['unit']==unit) & (df_loo['ID']==iloc), 'bb_loo_loc'].values
                aa_loo_var = df_loo.loc[(df_loo['feature']==feature) & (df_loo['unit']==unit) & (df_loo['ID']==iloc), 'aa_loo_loc_var'].values
                bb_loo_var = df_loo.loc[(df_loo['feature']==feature) & (df_loo['unit']==unit) & (df_loo['ID']==iloc), 'bb_loo_loc_var'].values
                y_pred = aa_loo*X + bb_loo*np.ones(np.shape(X))
                # y_pred = aa_loo*X_pred + bb_loo*np.ones(np.shape(X_pred))
                
                df_tmp['z_bsf'] = X.flatten()
                df_tmp['y_true'] = y.flatten()
                df_tmp['feature'] = feature
                df_tmp['unit'] = unit
                df_tmp['ID'] = iloc
                df_tmp.sort_values(by=['z_bsf'], inplace=True)  
                
                df_tmp_pred['z_bsf'] = X.flatten()
                df_tmp_pred['y_pred'] = y_pred.flatten()
                df_tmp_pred['y_pred_var'] = (aa_loo_var+bb_loo_var)*np.ones(np.shape(X)).flatten()
                df_tmp_pred.sort_values(by=['z_bsf'], inplace=True)  

                
                df_tmp_truepred = pd.merge_asof(df_tmp, df_tmp_pred, on=['z_bsf'])

                df_truepred = df_truepred.append(df_tmp_truepred, ignore_index=True)  

#%%
df_LL_n = pd.DataFrame([])
for feature in ['qc', 'fs', 'u2']:
    df_feat = df_truepred.loc[df_truepred['feature']==feature,:]
    for unit in df_feat['unit'].unique():
        # print(unit)
        df_unit = df_feat.loc[df_feat['unit']==unit,:]
        for loc in df_unit['ID'].unique():
            # print(loc)
            df_tmp = pd.DataFrame([])
            df_loc = df_unit.loc[df_unit['ID']==loc,:]
            n_pc = 0
            a = 0
            b = 5000
            res1 = 10
            res2 = b-a
            while (res1>0.01) and (res2>0.01):
                m = (a+b)/2
                B = df_loc['y_pred']-m*np.sqrt((df_loc['y_pred_var'])).values
                D = df_loc['y_pred']+m*np.sqrt((df_loc['y_pred_var'])).values
                is95 = (df_loc['y_true']<=D) & (df_loc['y_true']>=B)
                n_in = np.sum(is95)
                n_tot = len(is95)
                n_pc = n_in*100/n_tot
                if n_pc>=95:
                    b = m
                else:
                    a = m
                res1 = np.abs(n_pc-95)
                res2 = b-a
    
            print(unit, loc, a, n_pc)
            df_tmp.loc[0,'unit'] = unit
            df_tmp.loc[0,'ID'] = int(loc)
            df_tmp.loc[0,'feature'] = feature
            df_tmp.loc[0, 'n'] = m
            df_tmp.loc[0, 'n_pc'] = n_pc
            
            df_LL_n = df_LL_n.append(df_tmp, ignore_index=True)

#%%
for feature in ['qc', 'fs', 'u2']:
    df_feat = df_LL_n.loc[df_LL_n['feature']==feature,:]
    df_n = pd.DataFrame(columns=df_LL_n['unit'].sort_values().unique(), index=df_LL_n['ID'].sort_values().unique())
    for index, row in df_feat.iterrows():
        # print(row['ID'], row['unit'])
        df_n.loc[row['ID'], row['unit']] = row['n']
        
    #%
    n = df_n.to_numpy(dtype='float')
    if feature == 'qc':
        n_trans = RobustScaler(with_centering=False, quantile_range=(0, 70)).fit_transform(n)
    else:
        n_trans = np.copy(n)
    
    #% plot histogram
    fig, ax = plt.subplots()
    nhist, bins, _ = ax.hist(n_trans.flatten(), density=True, bins=50)
    plt.grid(True)
    ax.set_xlabel('n')
    ax.set_ylabel('Probability')
    ax.set_title('Calibration factor $n$ Histogram for $%s$' %(feature))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(np.max(bins)-np.max(bins)/4, np.max(nhist), 'mean = %.2f' % (np.nanmean(n_trans)),
             fontsize=12, verticalalignment='top', bbox=props)
    
    path_fig = '../../09-Results/Stage-03/LL-%s-Calibration_n-Histogram.png' %(feature)
    plt.savefig(path_fig, dpi=400)
    
    # plot n factor per unit per CPT
    y_ticklabel = df_n.columns.values
    x_ticklabel = df_n.index.values.astype(int)
    
    fig1, ax1 = plt.subplots(1,1,figsize=(32, 8))
    ax1.set_title('Calibration factor $n$, for $%s$' %(feature))
    c = ax1.pcolormesh(np.flipud(n_trans.T), 
                       norm=colors.LogNorm(vmin=np.nanmin(n_trans), vmax=np.nanmax(n_trans)),
                       edgecolors='k', linewidths=0.5)
    
    show_values(ax1, c)

    plt.yticks([])
    plt.xticks([])
    
    cell_text=[]
    for ii in range(n_trans.shape[1]):
        cell_text.append(['%1.1f' % np.nanmean(n_trans[:,ii])])
        
    left_table = plt.table(cellText=cell_text,
                          colLabels=['Mean'],
                          rowLabels=y_ticklabel,
                          cellLoc='center', 
                          loc='left',
                          bbox=[-0.019, 0.0, 0.019, 1+1/len(y_ticklabel)])
    
    cell_text=[]
    cell_text.append(['%1.1f' % x for x in np.nanmean(n_trans, axis=1)])
    bottom_table = plt.table(cellText=cell_text,
                          rowLabels=['Mean'],
                          colLabels=x_ticklabel,
                          cellLoc='center', 
                          loc='bottom')
    bottom_table.scale(1, 2)
   
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="1%", pad=0.05)   
    cbar = plt.colorbar(c, cax=cax)
    cbar.set_label('Calibration factor n')
    # tick_locator = ticker.MaxNLocator(nbins=10)
    # cbar.locator = tick_locator
    # cbar.update_ticks()
    
    path_fig = '../../09-Results/Stage-03/LL-%s-Calibration_n.png' %(feature)
    plt.savefig(path_fig, dpi=400)








