# -*- coding: utf-8 -*-
"""
Created on Tue May  4 15:19:09 2021

@author: GuS
"""

#%% Import libraries
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from scipy.stats import norm

import GM_Toolbox as GMT 

#%% Define functions
def load_obj(path_obj):
    with open(path_obj, 'rb') as f:
        return pickle.load(f)


def plot_regression_results(fig, ax, y_true, y_pred, feature, unit, scores=[]):
    model_dist, mae, accuracy, mu, std, mape = GMT.evaluate_modeldist_norm(y_true, y_pred)
    if not scores:
        scores_txt = (r'$MAE={:.2f}$'+ '\n' + r'$Accuracy={:.2f}$ %').format(mae, accuracy)
    else:
        mae, mae_std = scores[0], scores[1]
        scores_txt = (r'$MAE={:.2f} \pm {:.2f}$' + '\n' + r'$Accuracy={:.2f}$ %').format(mae, mae_std, accuracy)
        # scores_txt = (r'$MAE={:.2f} \pm {:.2f}$' + '\n').format(mae, mae_std, accuracy)
    """Scatter plot of the predicted vs true targets."""
    hb = ax.hexbin(y_true, y_pred, gridsize=25, bins='log', alpha=0.5, edgecolors=None,
                   extent=(y_true.min(), y_true.max(), y_true.min(), y_true.max()), cmap='afmhot_r')
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('counts')   
    # ax.plot(y_true, y_pred, '.', alpha=0.2,  color='tab:blue', markersize=5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],'--k', linewidth=2)
    ax.plot([y_true.min(), y_true.max()], [(1+std)*y_true.min(), (1+std)*y_true.max()],':k', linewidth=2)
    ax.plot([y_true.min(), y_true.max()], [(1-std)*y_true.min(), (1-std)*y_true.max()],':k', linewidth=2)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # ax.set_xlim([y_pred.min(), y_pred.max()])
    # ax.set_ylim([y_pred.min(), y_pred.max()])
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
    n, bins, patches = ax.hist(model_dist, 50, edgecolor='black', density=True, range=[-2.5, 2.5], facecolor='green', alpha=0.5)
    x = np.linspace(-2.5, 2.5, 100)
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

def plot_global_score(df_RF_reg, df_RF_scores):
    # Plot global scores
    fig, axs = plt.subplots(3, 2, figsize=(14,14))
    fig.suptitle('3D UKriging - Global scores', fontsize=14)
    unit = 'global'
    for ii, feature in zip([0,1,2], ['qc', 'fs', 'u2']):
        # [mae] = df_RF_scores.loc[df_RF_scores['feature']==feature, 'mae_kfold'].values
        # [mae_std] = df_RF_scores.loc[df_RF_scores['feature']==feature, 'mae_kfold_std'].values   
        # scores = [mae, mae_std]
        y_true = df_RF_reg.loc[df_RF_reg['feature']==feature, feature]
        y_pred = df_RF_reg.loc[df_RF_reg['feature']==feature, 'y_pred_loo']
        axs[ii,0] = plot_regression_results(fig, axs[ii,0], y_true, y_pred, feature, unit)
        axs[ii,1] = plot_histogram_results(axs[ii,1], y_true, y_pred, feature, unit)  
    plt.tight_layout()
    return fig, axs
    
def plot_unit_score(df_RF_reg, pdf):
    # Plot scores per unit
    # unitlist = df_RF_reg['unit_geo'].dropna().unique()
    unitlist = df_RF_reg['unit'].dropna().unique()
    for unit in unitlist:
        print('unit: ' + unit)
        # setup figure and axs
        fig, axs = plt.subplots(3, 2, figsize=(14,14))
        fig.suptitle('3D UKriging - Unit ' + unit + ' scores', fontsize=14)
        for ii, feature in zip([0,1,2], ['qc', 'fs', 'u2']):
            y_true = df_RF_reg.loc[(df_RF_reg['feature']==feature) & (df_RF_reg['unit']==unit), feature]
            y_pred = df_RF_reg.loc[(df_RF_reg['feature']==feature) & (df_RF_reg['unit']==unit), 'y_pred_loo']
            axs[ii,0] = plot_regression_results(fig, axs[ii,0], y_true, y_pred, feature, unit)
            axs[ii,1] = plot_histogram_results(axs[ii,1], y_true, y_pred, feature, unit)  
        plt.tight_layout()
        pdf.savefig(fig)
    return pdf



#%% Load database
path_database = '../../09-Results/Stage-01/Database.pkl'
df = pd.read_pickle(path_database)
# Limit depth
z_min, z_max = -1, 80
# z_min, z_max = df['z_bsf'].min(), df['z_bsf'].max()
df = df[(df['z_bsf']>=z_min) & (df['z_bsf']<=z_max)]

#%% Load RFreg LeaveOneOut and KFold results
path_dict = '../../09-Results/Stage-01/UKriging3D-Kfold-Scores.pkl'
df_UK3D_scores = load_obj(path_dict)
path_dict = '../../09-Results/Stage-01/UKriging3D-Kfold-Results.pkl'
df_UK3D_reg = load_obj(path_dict)

#%% Main
path_pdf = '../../09-Results/Stage-01/3DKriging-Scores.pdf'
pdf = PdfPages(path_pdf)
fig, axs = plot_global_score(df_UK3D_reg, df_UK3D_scores)
pdf.savefig(fig)
pdf = plot_unit_score(df_UK3D_reg, pdf)
pdf.close()  












