# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 13:36:21 2022

@author: GuS
"""


#%% Import libraries
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import interpolate

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

def load_RF_mev_ann(path_dir_pred):
    df_qc = pd.DataFrame([])
    df_fs = pd.DataFrame([])
    df_u2 = pd.DataFrame([])
    for filename in os.listdir(path_dir_pred):        
        if filename.endswith("qc.range"):
            df = pd.read_csv(os.path.join(path_dir_pred, filename), sep='\s+',
                            header=None, names=['z_bsf', 'qc_be', 'qc_min', 'qc_max'],
                            index_col=False, skiprows=1)
            df['ID'] = int(''.join(filter(lambda i: i.isdigit(), filename)))
            df_qc = df_qc.append(df)
        elif filename.endswith("fs.range"):
            df = pd.read_csv(os.path.join(path_dir_pred, filename), sep='\s+',
                            header=None, names=['z_bsf', 'fs_be', 'fs_min', 'fs_max'],
                            index_col=False, skiprows=1)
            df['ID'] = int(''.join(filter(lambda i: i.isdigit(), filename)))
            df_fs = df_fs.append(df)
        elif filename.endswith("u2.range"):
            df = pd.read_csv(os.path.join(path_dir_pred, filename), sep='\s+',
                            header=None, names=['z_bsf', 'u2_be', 'u2_min', 'u2_max'],
                            index_col=False, skiprows=1)
            df['ID'] = int(''.join(filter(lambda i: i.isdigit(), filename[:-9])))
            df_u2 = df_u2.append(df)
    df_mev = df_qc.merge(df_fs, how='outer', on=['z_bsf', 'ID'])
    df_mev = df_mev.merge(df_u2, how='outer', on=['z_bsf', 'ID'])
    return df_mev

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
#%% Load mean database
z_min, z_max = -1, 80
path_mean = '../../09-Results/Stage-01/Database_mean.csv'
df_mean = pd.read_csv(path_mean)
df_mean = df_mean[(df_mean['z_bsf']>=z_min) & (df_mean['z_bsf']<=z_max)]


#%% Load AI-ANN MEV
path_dir_ann = '../../09-Results/Stage-01/FromMark/AI_ANN/Ranges'
df_mev_ann = load_RF_mev_ann(path_dir_ann)

#%% Main loop over CPT location (ID)
# List of locations and units
loclist = df_mev_ann['ID'].sort_values().unique()

df_truepred = pd.DataFrame([])
for loc in loclist:
    df_tmp = pd.DataFrame([])
    iloc = int(loc)
    print('Loc:', iloc)
    df_true_loc = df_mean.loc[df_mean['ID']==loc, ['z_bsf', 'unit', 'qc', 'fs', 'u2', 'ID']]
    df_pred_loc = df_mev_ann.loc[df_mev_ann['ID']==loc,:]
    df_tmp = df_true_loc.copy()
    
    for colname in ['qc_be', 'qc_min', 'qc_max', 'fs_be', 'fs_min', 'fs_max', 'u2_be', 'u2_min', 'u2_max']:
        f = interpolate.interp1d(df_pred_loc['z_bsf'], df_pred_loc[colname], 
                                 bounds_error=False, fill_value=np.nan)
        y_int = f(df_true_loc['z_bsf'])
        df_tmp[colname] = y_int
    # df_tmp = pd.merge_asof(df_pred_loc, df_true_loc, on=['z_bsf'])
    
    df_truepred = df_truepred.append(df_tmp, ignore_index=True)  
    

#%%
df_AIANN_n = pd.DataFrame([])
for feature in ['qc', 'fs', 'u2']:
    coln_b = feature + '_min'
    coln_c = feature + '_be'
    coln_d = feature + '_max'
    coln_true = feature
    for unit in df_truepred['unit'].unique():
    # for unit in ['GGM61']:
        # print(unit)
        df_unit = df_truepred.loc[df_truepred['unit']==unit,:].copy()
        for loc in df_unit['ID'].unique():
        # for loc in [5]:
            # print(loc)
            df_tmp = pd.DataFrame([])
            df_loc = df_unit.loc[df_unit['ID']==loc,:].copy()
            df_loc.dropna(subset=[feature], inplace=True)
            if not df_loc.empty:
                n_pc = 0
                a = 0
                b = 20
                res1 = 10
                res2 = b-a
                m=0
                while (res1>0.1) and (res2>0.0001):
                    m = (a+b)/2
                    B = df_loc[coln_b].values-m*df_loc[coln_b].values*np.sign(df_loc[coln_b].values)
                    D = df_loc[coln_d].values+m*df_loc[coln_b].values*np.sign(df_loc[coln_b].values)
                    # B = -m+df_loc[coln_b].values
                    # D = m+df_loc[coln_d].values
                    C = df_loc[coln_c].values
                    y_true = df_loc[coln_true].values
                    is95 = (y_true<=D) & (y_true>=B)
                    n_in = np.sum(is95)
                    n_tot = len(is95)
                    n_pc = n_in*100/n_tot
                    if n_pc>=95:
                        b = m
                    else:
                        a = m
                    res1 = np.abs(n_pc-95)
                    res2 = b-a
                    # print(m, res1, res2, n_pc)
                    # plt.plot(B,'r', label='B'); plt.plot(D,'b', label='D'); plt.plot(y_true,'k',label='true')
                    # plt.legend()
                    # plt.show()
                
                print('%s, %i, n: %.2f, n_pc: %.2f' %(unit, loc, m, n_pc))
                
                df_tmp.loc[0,'unit'] = unit
                df_tmp.loc[0,'ID'] = int(loc)
                df_tmp.loc[0,'feature'] = feature
                df_tmp.loc[0, 'n'] = m
                df_tmp.loc[0, 'n_pc'] = n_pc
                
                df_AIANN_n = df_AIANN_n.append(df_tmp, ignore_index=True)

#%%
for feature in ['qc', 'fs', 'u2']:
    df_feat = df_AIANN_n.loc[df_AIANN_n['feature']==feature,:]
    df_n = pd.DataFrame(columns=df_AIANN_n['unit'].sort_values().unique(), index=df_AIANN_n['ID'].sort_values().unique())
    for index, row in df_feat.iterrows():
        # print(row['ID'], row['unit'])
        df_n.loc[row['ID'], row['unit']] = row['n']
        
    #%
    n = df_n.to_numpy(dtype='float')
    if feature == 'qc':
        n_trans = np.copy(n)
        n_trans[n_trans>14.5] = np.nan
    #     n_trans = RobustScaler(with_centering=False, quantile_range=(0, 80)).fit_transform(n)    
    else:
        n_trans = np.copy(n)
        n_trans[n_trans>19.5] = np.nan
        
    
    #% plot histogram
    fig, ax = plt.subplots()
    nhist, bins, _ = ax.hist(n_trans.flatten(), density=True, bins=100)
    plt.grid(True)
    ax.set_xlabel('n')
    ax.set_ylabel('Probability')
    ax.set_title('AIANN - Calibration factor $n$ Histogram for $%s$' %(feature))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(np.max(bins)-np.max(bins)/4, np.max(nhist), 'mean = %.2f' % (np.nanmean(n_trans)),
             fontsize=12, verticalalignment='top', bbox=props)
    
    path_fig = '../../09-Results/Stage-03/99-Calibration_n/AIANN-%s-Calibration_n-Histogram.png' %(feature)
    plt.savefig(path_fig, dpi=400)
    
    # plot n factor per unit per CPT
    y_ticklabel = df_n.columns.values
    x_ticklabel = df_n.index.values.astype(int)
    
    fig1, ax1 = plt.subplots(1,1,figsize=(32, 8))
    ax1.set_title('AIANN - Calibration factor $n$, for $%s$' %(feature))
    c = ax1.pcolormesh(np.flipud(n_trans.T), 
                        norm=colors.LogNorm(vmin=np.nanmin(n_trans), vmax=np.nanmax(n_trans)),
                        edgecolors='k', linewidths=0.5)
    
    show_values(ax1, c)

    plt.yticks([])
    plt.xticks([])
    
    cell_text=[]
    for ii in range(n_trans.shape[1]):
        cell_text.append(['%1.1f' % np.nanmean(n_trans[:,ii])])
        
    left_table = ax1.table(cellText=cell_text,
                          colLabels=['Mean'],
                          rowLabels=y_ticklabel,
                          cellLoc='center', 
                          loc='left',
                          bbox=[-0.019, 0.0, 0.019, 1+1/len(y_ticklabel)])
    
    cell_text=[]
    cell_text.append(['%1.1f' % x for x in np.nanmean(n_trans, axis=1)])
    bottom_table = ax1.table(cellText=cell_text,
                          rowLabels=['Mean'],
                          colLabels=x_ticklabel,
                          cellLoc='center', 
                          loc='bottom')
    bottom_table.scale(1, 2)
   
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="1%", pad=0.05)   
    cbar = plt.colorbar(c, cax=cax)
    cbar.set_label('Calibration factor n')

    # fig.tight_layout()
    path_fig = '../../09-Results/Stage-03/99-Calibration_n/AIANN-%s-Calibration_n.png' %(feature)
    plt.savefig(path_fig, dpi=400)








