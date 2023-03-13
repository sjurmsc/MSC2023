# -*- coding: utf-8 -*-
"""
Created on Wed May  5 22:57:26 2021

@author: GuS
"""


#%% Import libraries
import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import GM_Toolbox as GMT 


#%% Define functions
def load_obj(path_obj):
    with open(path_obj, 'rb') as f:
        return pickle.load(f)
    
def load_RF_mev(path_dir_pred):
    df_qc = pd.DataFrame([])
    df_fs = pd.DataFrame([])
    df_u2 = pd.DataFrame([])
    for filename in os.listdir(path_dir_pred):        
        if filename.endswith("qc.range"):
            df = pd.read_csv(os.path.join(path_dir_pred, filename), sep='\s+',
                            header=None, names=['z_bsf', 'qc_min', 'qc_max'],
                            index_col=False, skiprows=1)
            df['ID'] = int(''.join(filter(lambda i: i.isdigit(), filename)))
            df_qc = df_qc.append(df)
        elif filename.endswith("fs.range"):
            df = pd.read_csv(os.path.join(path_dir_pred, filename), sep='\s+',
                            header=None, names=['z_bsf', 'fs_min', 'fs_max'],
                            index_col=False, skiprows=1)
            df['ID'] = int(''.join(filter(lambda i: i.isdigit(), filename)))
            df_fs = df_fs.append(df)
        elif filename.endswith("u2.range"):
            df = pd.read_csv(os.path.join(path_dir_pred, filename), sep='\s+',
                            header=None, names=['z_bsf', 'u2_min', 'u2_max'],
                            index_col=False, skiprows=1)
            df['ID'] = int(''.join(filter(lambda i: i.isdigit(), filename[:-9])))
            df_u2 = df_u2.append(df)
    df_mev = df_qc.merge(df_fs, how='outer', on=['z_bsf', 'ID'])
    df_mev = df_mev.merge(df_u2, how='outer', on=['z_bsf', 'ID'])
    return df_mev

def plot_regression_results(fig, ax, y_true, y_pred, feature, unit, scores=[]):
    model_dist, mae, accuracy, mu, std, mape = GMT.evaluate_modeldist_norm(y_true, y_pred)
    if not scores:
        scores_txt = (r'$MAE={:.2f}$' + '\n' + r'$Error={:.2f}$ %').format(mae, accuracy)
        # scores_txt = (r'$MAE={:.2f}$').format(mae)
    else:
        mae, mae_std = scores[0], scores[1]
        scores_txt = (r'$MAE={:.2f} \pm {:.2f}$' + '\n' + r'$Accuracy={:.2f}$ %').format(mae, mae_std, mape)
        # scores_txt = (r'$MAE={:.2f} \pm {:.2f}$' + '\n' + r'$Accuracy={:.2f}$ %').format(mae, mae_std, accuracy)
        # scores_txt = ('\n' + r'$MAE={:.2f} \pm {:.2f}$').format(mae, mae_std)
    """Scatter plot of the predicted vs true targets."""
    hb = ax.hexbin(y_true, y_pred, gridsize=25, bins='log', alpha=0.5, edgecolors=None,
                   extent=(y_true.min(), y_true.max(), y_true.min(), y_true.max()), cmap='afmhot_r')
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('counts')   
    # ax.plot(y_true, y_pred, '.', alpha=0.2,  color='tab:blue', markersize=5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],'--k', linewidth=2)
    # ax.plot([y_true.min(), y_true.max()], [(1+std)*y_true.min(), (1+std)*y_true.max()],':k', linewidth=2)
    # ax.plot([y_true.min(), y_true.max()], [(1-std)*y_true.min(), (1-std)*y_true.max()],':k', linewidth=2)
    ax.plot([y_true.min(), y_true.max()], [(1+0.3)*y_true.min(), (1+0.3)*y_true.max()],':k', linewidth=2)
    ax.plot([y_true.min(), y_true.max()], [(1-0.3)*y_true.min(), (1-0.3)*y_true.max()],':k', linewidth=2)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # ax.set_xlim([y_pred.min(), y_pred.max()])
    # ax.set_ylim([y_pred.min(), y_pred.max()])
    if feature == 'qc':
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 100])
    ax.set_xlabel('Measured $%s_{%s}$' %(feature[0], feature[1]))
    ax.set_ylabel('Predicted $%s_{%s}$' %(feature[0], feature[1]))
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                          edgecolor='none', linewidth=0)
    ax.legend([extra], [scores_txt], loc='upper left', prop={'size': 10})
    title = unit + ' - Cross-validation'
    ax.set_title(title)
    ax.grid(True)
    return ax
    
def plot_histogram_results(ax, y_true, y_pred, feature, unit):
    model_dist, mae, accuracy, mu, std, mape = GMT.evaluate_modeldist_norm(y_true, y_pred)
    # n, bins, patches = ax.hist(model_dist, 50, edgecolor='black', density=True, stacked=True, range=[-2.50, 2.50], facecolor='green', alpha=0.5)
    # x = np.linspace(-2.50, 2.50, 100)
    # weights = np.ones(len(model_dist))/len(model_dist)
    n, bins, p = ax.hist(model_dist, 50, edgecolor='black', density=True,
                         facecolor='green', alpha=0.5, align='mid')
    x = np.linspace(np.min(model_dist), np.max(model_dist), 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2)
    ax.set_xlabel('$(%s_{%s_{pred}}-%s_{%s})/%s_{%s}$'%(feature[0], feature[1],feature[0], feature[1],feature[0], feature[1]))
    # ax.set_xlabel('$%s_{%s_{pred}}-%s_{%s}$'%(feature[0], feature[1], feature[0], feature[1]))
   
    ax.set_ylabel('Probability')
    ax.set_title(unit + ' - Histogram')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                          edgecolor='none', linewidth=0)
    ax.legend([extra], ['$\mu=%.3f$\n$\sigma=%.3f$' %(mu, std)], loc='upper left', prop={'size': 10})
    ax.grid(True)
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
    
    
#%% Load prediction results
#% Load database
# path_database = '../../09-Results/Stage-01/Database.pkl'
# df = pd.read_pickle(path_database)
path_database = '../../09-Results/Stage-01/Database_mean.pkl'
df = pd.read_pickle(path_database)
# Limit depth
z_min, z_max = -1, 80
df = df[(df['z_bsf']>=z_min) & (df['z_bsf']<=z_max)]

#%% Load Local Linear Regression LeaveOneOut and KFold results
path_dict = '../../09-Results/Stage-01/LocalLinearFit-UKriging-GroupKfold-Scores.pkl'
df_LL_scores = load_obj(path_dict)
path_dict = '../../09-Results/Stage-01/LocalLinearFit-UKriging-GroupKfold-Results.pkl'
df_LL_res = load_obj(path_dict)

#%% Load RFreg LeaveOneOut and KFold results
path_dict = '../../09-Results/Stage-01/RandomForest-Kfold-Scores.pkl'
df_RF_scores = load_obj(path_dict)
path_dict = '../../09-Results/Stage-01/RandomForest-Kfold-Results.pkl'
df_RF_res = load_obj(path_dict)

#%% Load Rf classificationb from Mark
path_dir_pred = '../../09-Results/Stage-01/FromMark/Ranges'
df_mev = load_RF_mev(path_dir_pred)

#%% Load Global linear regression
path_dict = '../../09-Results/Stage-01/GlobalLinearFit-Kfold-Results.pkl'
df_GL_res = load_obj(path_dict)  

#%% Load UK3D
path_dict = '../../09-Results/Stage-01/UKriging3D-Kfold-Results.pkl'
df_UK3D_res = load_obj(path_dict)  

#%% Load AIANN MEV
path_dir_ann = '../../09-Results/Stage-01/FromMark/AI + ANN/Ranges'
df_mev_ann = load_RF_mev_ann(path_dir_ann)


#%% Global results comparison
loclist = df['ID'].unique()
model_list = ['Global Linear', 'Local Linear', 'UK3D', 'RF Regression']
path_pdf = '../../09-Results/Stage-01/ModelComparison-Scores.pdf'
pdf = PdfPages(path_pdf)
for feature in ['qc', 'fs', 'u2']:
    print('Feature: ', feature)
    fig, axs = plt.subplots(len(model_list), 2, figsize=(14,14))
    fig.suptitle('Predictive model comparison - ' + feature + ' - Global scores', fontsize=14)
    for ii, model in zip(np.arange(0, len(model_list), 1), model_list):
        print('\t',model)
        if model == 'Global Linear':
            # Loop over units
            df_gl_pred = pd.DataFrame([])
            for unit in df['unit'].dropna().unique():
                df_tmp = pd.DataFrame([])
                df_data = df.loc[df['unit']==unit, ['ID','z_bsf', 'x','y', 'qc', 'fs', 'u2']].dropna(subset=[feature])               
                X = df_data['z_bsf'].values.reshape(-1,1)
                y = df_data[feature].values.reshape(-1,1)
            
                [slope] = df_GL_res.loc[(df_GL_res['feature']==feature) & (df_GL_res['unit_geo']==unit),'slope'].values
                [intercept] = df_GL_res.loc[(df_GL_res['feature']==feature) & (df_GL_res['unit_geo']==unit),'intercept'].values
                df_tmp['X'] = X.flatten()
                df_tmp['y'] = y
                df_tmp['y_pred'] = slope*X + intercept*np.ones(np.shape(X))
                df_gl_pred = df_gl_pred.append(df_tmp)
            
            unit = 'Global ' + model
            axs[ii,0] = plot_regression_results(fig, axs[ii,0], df_gl_pred['y'].values, df_gl_pred['y_pred'].values, feature, unit, scores=[])
            axs[ii,1] = plot_histogram_results(axs[ii,1], df_gl_pred['y'], df_gl_pred['y_pred'], feature, unit)
            
        if model == 'Local Linear':
            # Loop over locations
            df_ll_pred = pd.DataFrame([])
            for loc in loclist:
                df_loc = df.loc[df['ID']==loc,:]
                # Loop over units
                for unit in df['unit'].dropna().unique():
                    df_tmp = pd.DataFrame([])
                    df_data = df_loc.loc[df_loc['unit']==unit,['z_bsf', feature]].dropna()             
                    X = df_data['z_bsf'].values.reshape(-1,1)
                    y = df_data[feature].values.reshape(-1,1)
                    
                    slope = df_LL_res.loc[(df_LL_res['feature']==feature) & (df_LL_res['ID']==loc) & (df_LL_res['unit']==unit),'slope'].values
                    intercept = df_LL_res.loc[(df_LL_res['feature']==feature) & (df_LL_res['ID']==loc) & (df_LL_res['unit']==unit),'intercept'].values
                    
                    if slope.size>0:
                        y_pred = slope*X + intercept*np.ones(np.shape(X))
                    else:
                        [UK_slope] = df_LL_scores.loc[(df_LL_scores['unit']==unit) & (df_LL_scores['feature']==feature), 'slope_kri_obj']
                        [UK_intercept] = df_LL_scores.loc[(df_LL_scores['unit']==unit) & (df_LL_scores['feature']==feature), 'intercept_kri_obj']
                        if isinstance(UK_slope, float):
                            slope = UK_slope
                        else:
                            slope, _ = UK_slope.execute('points',
                                                        df_loc.loc[df_loc['unit']==unit,['x']].drop_duplicates().mean().values,
                                                        df_loc.loc[df_loc['unit']==unit,['y']].drop_duplicates().mean().values)
                        if isinstance(UK_intercept, float):
                            intercept = UK_intercept
                        else:
                            intercept, _ = UK_intercept.execute('points',
                                                        df_loc.loc[df_loc['unit']==unit,['x']].drop_duplicates().mean().values,
                                                        df_loc.loc[df_loc['unit']==unit,['y']].drop_duplicates().mean().values)
                        y_pred = slope* + intercept*np.ones(np.shape(X))
                    
                    df_tmp['X'] = X.flatten()
                    df_tmp['y'] = y
                    df_tmp['y_pred'] = y_pred
                    df_ll_pred = df_ll_pred.append(df_tmp)
    
            mae = df_LL_scores.loc[(df_LL_scores['feature']==feature),'slope_mae'].mean()
            mae_std = df_LL_scores.loc[(df_LL_scores['feature']==feature),'slope_mae_std'].mean()
            scores = [mae, mae_std]
            unit = 'Global ' + model
            axs[ii,0] = plot_regression_results(fig, axs[ii,0], df_ll_pred['y'], df_ll_pred['y_pred'], feature, unit, scores=[])
            axs[ii,1] = plot_histogram_results(axs[ii,1], df_ll_pred['y'], df_ll_pred['y_pred'], feature, unit)
            
        if model == 'UK3D':
            y_true = df_UK3D_res.loc[df_UK3D_res['feature']==feature, feature]
            y_pred = df_UK3D_res.loc[df_UK3D_res['feature']==feature, 'y_pred_loo']
            unit = 'Global ' + model
            axs[ii,0] = plot_regression_results(fig, axs[ii,0], y_true, y_pred, feature, unit, scores=[])
            axs[ii,1] = plot_histogram_results(axs[ii,1], y_true, y_pred, feature, unit)

        if model == 'RF Regression':
            [mae] = df_RF_scores.loc[df_RF_scores['feature']==feature, 'mae_kfold'].values
            [mae_std] = df_RF_scores.loc[df_RF_scores['feature']==feature, 'mae_kfold_std'].values   
            scores = [mae, mae_std]
            y_true = df_RF_res.loc[df_RF_res['feature']==feature, feature]
            y_pred = df_RF_res.loc[df_RF_res['feature']==feature, 'y_pred_loo']
            unit = 'Global ' + model
            axs[ii,0] = plot_regression_results(fig, axs[ii,0], y_true, y_pred, feature, unit, scores=[])
            axs[ii,1] = plot_histogram_results(axs[ii,1], y_true, y_pred, feature, unit)
            
    plt.tight_layout()
    pdf.savefig(fig)
pdf.close()














