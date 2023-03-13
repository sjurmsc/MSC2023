# -*- coding: utf-8 -*-
"""
Created on Wed May  5 22:56:39 2021

@author: GuS
"""

#%% Import libraries
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

import GM_Toolbox as GMT 

#%% Define functions
def data_preparation(y,y_pred=0,detrend=0,bounds=[-1, 1]):
    if detrend=='quartile':
        errors = y_pred - y
        q25,q75=np.percentile(errors,[25,75])
        intqrt=q75-q25
        y0=y[(errors>(q25-1.5*intqrt)) & (errors<(q75+1.5*intqrt))]
        y_pred0=y_pred[(errors>(q25-1.5*intqrt)) & (errors<(q75+1.5*intqrt))]
        return y0, y_pred0

    
def plot_regression_results(fig, ax, y_true, y_pred, feature, unit, scores=[]):
    y_pred=y_pred.flatten()
    y_true=y_true.flatten()

    # y_true0, y_pred0 = data_preparation(y_true,y_pred,detrend='quartile',bounds=[-1, 1])
    
    model_dist, mae, accuracy, mu, std, mape = GMT.evaluate_modeldist(y_true, y_pred)
    
    if not scores:
        # scores_txt = (r'$MAE={:.2f}$'+ '\n' + r'$Accuracy={:.2f}$ %').format(mae, accuracy)
        scores_txt = (r'$MAE={:.2f}$ [MPa]'+ '\n').format(mae)
    else:
        mae, mae_std = scores[0], scores[1]
        scores_txt = (r'$MAE={:.2f} \pm {:.2f}$ [MPa]' + '\n').format(mae, mae_std)
        # scores_txt = (r'$MAE={:.2f} \pm {:.2f}$' + '\n' + r'$Accuracy={:.2f}$ %').format(mae, mae_std, accuracy)
        # scores_txt = (r'$MAE={:.2f} \pm {:.2f}$' + '\n').format(mae, mae_std, accuracy)
    """Scatter plot of the predicted vs true targets."""
    hb = ax.hexbin(y_true, y_pred, gridsize=25, bins='log', alpha=0.5, edgecolors=None,
                   extent=(y_true.min(), y_true.max(), y_true.min(), y_true.max()), cmap='afmhot_r')
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('counts')   
    # ax.plot(y_true, y_pred, '.', alpha=0.2,  color='tab:blue', markersize=5)
    
    # ax.plot([y_pred.min(), y_pred.max()], [y_pred.min(), y_pred.max()],'--k', linewidth=2)
    # ax.plot([y_pred.min()+std, y_pred.max()+std], [y_pred.min(), y_pred.max()],':k', linewidth=2)
    # ax.plot([y_pred.min()-std, y_pred.max()-std], [y_pred.min(), y_pred.max()],':k', linewidth=2)
    
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],'--k', linewidth=2)
    ax.plot([y_true.min()+std, y_true.max()+std], [y_true.min(), y_true.max()],':k', linewidth=2)
    ax.plot([y_true.min()-std, y_true.max()-std], [y_true.min(), y_true.max()],':k', linewidth=2)
    
    # ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],'--k', linewidth=2)
    # ax.plot([y_true.min(), y_true.max()], [(1+std)*y_true.min(), (1+std)*y_true.max()],':k', linewidth=2)
    # ax.plot([y_true.min(), y_true.max()], [(1-std)*y_true.min(), (1-std)*y_true.max()],':k', linewidth=2)
    
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
    y_true=y_true.flatten()
    y_pred=y_pred.flatten()
    # y_true0,y_pred0=data_preparation(y_true,y_pred,detrend='quartile',bounds=[-1, 1])
    # model_dist, mae, accuracy, mu, std, mape = GMT.evaluate_modeldist_norm(y_true0, y_pred0)
    model_dist, mae, accuracy, mu, std, mape = GMT.evaluate_modeldist(y_true, y_pred)
    n, bins, patches = ax.hist(y_pred - y_true, 50, edgecolor='black', density=True, range=[-5*std, 5*std], facecolor='green', alpha=0.5)
    
    # n, bins, patches = ax.hist(model_dist, 50, edgecolor='black', density=True, range=[-2.5, 2.5], facecolor='green', alpha=0.5)
    # x = np.linspace(-2.5, 2.5, 100)
    # p = norm.pdf(x, mu, std)
    # ax.plot(x, p, 'k', linewidth=2)
    # ax.set_xlabel('$(%s_{%s_{pred}}-%s_{%s})/%s_{%s}$'%(feature[0], feature[1],feature[0], feature[1],feature[0], feature[1]))
    # # ax.set_xlabel('$%s_{%s_{pred}}-%s_{%s}$'%(feature[0], feature[1], feature[0], feature[1]))
    # ax.set_ylabel('Probability')
    # ax.set_title(unit + ' - Histogram')
    # extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
    #                       edgecolor='none', linewidth=0)
    # ax.legend([extra], ['$\mu=%.3f$\n$\sigma=%.3f$' %(mu, std)], loc='upper left', prop={'size': 10})
    # ax.grid(True)
    
    x = np.linspace(-5*std, 5*std, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2)
    # ax.set_xlabel('$(%s_{%s_{pred}}-%s_{%s})/%s_{%s}$'%(feature[0], feature[1],feature[0], feature[1],feature[0], feature[1]))
    ax.set_xlabel('$%s_{%s_{pred}}-%s_{%s}$'%(feature[0], feature[1], feature[0], feature[1]))
    ax.set_ylabel('Probability')
    ax.set_title('%s - Histogram of $%s_{%s}$' %(unit, feature[0], feature[1]))
    extra = plt.Rectangle((0, 0), 0, 0,fc="w",fill=False,edgecolor='none',linewidth=0)
    
    # ratio=np.round(len(y[ (y_true>y_pred-std) & (y_true<y_pred+std) ])/len(y)*100,2)
    # ax.legend([extra],['$\mu=%.3f$\n$\sigma=%.3f$\n$R=%.2f$' %(mu,std,ratio)],loc='upper left',prop={'size':10})
    ax.legend([extra],['$\mu=%.3f$\n$\sigma=%.3f$ [MPa]' %(mu,std)],loc='upper left',prop={'size':10})
    ax.grid(True)
    
    labels=['-4*$\sigma$','-3*$\sigma$','-2*$\sigma$','-$\sigma$',0,'$\sigma$','2*$\sigma$','3*$\sigma$','4*$\sigma$']
    ax.set_xticks([-4*std, -3*std, -2*std, -std, 0, std, 2*std, 3*std, 4*std])
    ax.set_xticklabels(labels)
    return ax, std, mu, mae

# def plot_global_score(df_results, df_scores, model):
#     # Plot global scores
#     fig, axs = plt.subplots(3, 2, figsize=(14,14))
#     fig.suptitle(model + ' - Global scores', fontsize=14)
#     unit = 'global'
#     for ii, feature in zip([0,1,2], ['qc', 'fs', 'u2']):
#         # [mae] = df_scores.loc[df_scores['feature']==feature, 'mae_kfold'].values
#         # [mae_std] = df_scores.loc[df_scores['feature']==feature, 'mae_kfold_std'].values   
#         # scores = [mae, mae_std]
#         y_true = df_results.loc[df_results['feature']==feature, feature]
#         y_pred = df_results.loc[df_results['feature']==feature, 'y_pred_loo']
#         axs[ii,0] = plot_regression_results(fig, axs[ii,0], y_true, y_pred, feature, unit)
#         axs[ii,1] = plot_histogram_results(axs[ii,1], y_true, y_pred, feature, unit)  
#     plt.tight_layout()
#     return fig, axs
    
# def plot_unit_score(df_results, model, path_pngdir):
#     # Plot scores per unit
#     # unitlist = df_results['unit_geo'].dropna().unique()
#     unitlist = df_results['unit'].dropna().unique()
#     for unit in unitlist:
#         print('unit: ' + unit)
#         # path png
#         path_png = path_pngdir + model + '-Scores-Unit_%s.png' %(unit)
#         # setup figure and axs
#         fig, axs = plt.subplots(3, 2, figsize=(14,14))
#         fig.suptitle(model + ' - Unit ' + unit + ' scores', fontsize=14)
#         for ii, feature in zip([0,1,2], ['qc', 'fs', 'u2']):
#             y_true = df_results.loc[(df_results['feature']==feature) & (df_results['unit']==unit), feature]
#             y_pred = df_results.loc[(df_results['feature']==feature) & (df_results['unit']==unit), 'y_pred_loo']
#             axs[ii,0] = plot_regression_results(fig, axs[ii,0], y_true, y_pred, feature, unit)
#             axs[ii,1] = plot_histogram_results(axs[ii,1], y_true, y_pred, feature, unit)  
#         plt.tight_layout()
#         plt.savefig(path_png, dpi=200)


### Main ###
#%% Load data
print("Load xls with results")
pathdir = "../../09-Results/Stage-01/Stage-01_ModelComparison_03/"
df = pd.DataFrame() 
for filename in os.listdir(pathdir):
    if filename.endswith(".xlsx"):
        loc = filename[-8:-5]
        print(loc)
        df_tmp = pd.read_excel(os.path.join(pathdir, filename))
        df = df.append(df_tmp, ignore_index=True)
print("Loaded \n")
df["RFclass_mean_qc"] = df[["RF_class_min_qc", "RF_class_max_qc"]].mean(axis=1)
df["RFclass_mean_fs"] = df[["RF_class_min_fs", "RF_class_max_fs"]].mean(axis=1)
df["RFclass_mean_u2"] = df[["RF_class_min_u2", "RF_class_max_u2"]].mean(axis=1)

unitlist = df['unit'].dropna().unique()
modellist = ["GL", "LL", "UK3D", "RFreg", "RFclass_mean", "AIANN_best" ]
# modellist = ["UK3D", "RFreg", "RFclass_mean", "AIANN_best"]

df_scores = pd.DataFrame()

#%% Plot score global
print('Plot global score')
for model in modellist:
    unit = 'Global'
    path_png = '../../09-Results/Stage-01/Final/' + model + '-Scores_Units_All.png'
    fig, axs = plt.subplots(3, 2, figsize=(14,14))
    fig.suptitle(model + ' - Global scores', fontsize=14)
    for ii, param in zip([0,1,2], ['qc', 'fs', 'u2']):
        print(model, param)
        modpar = model + "_" + param
        df_results = df[[param, modpar]].dropna()
        y_true = df_results[param].values
        y_pred = df_results[modpar].values
        if y_pred.size>0:
            axs[ii,0] = plot_regression_results(fig, axs[ii,0], y_true, y_pred, param, unit)
            axs[ii,1], std, mu, mae = plot_histogram_results(axs[ii,1], y_true, y_pred, param, unit) 
            df_tmp = pd.DataFrame(data={'param': [param], 'model': [model], 'unit': ['all'], 'std': [std], 'mu': [mu], 'mae': [mae]})
            df_scores = df_scores.append(df_tmp)
    plt.tight_layout()
    plt.savefig(path_png, dpi=200)


#% Plot score per unit
print("Plot score per unit")
for model in modellist:
    for unit in unitlist:
        path_png = '../../09-Results/Stage-01/Final/' + model + '-Scores-Units_' + unit + '.png'
        fig, axs = plt.subplots(3, 2, figsize=(14,14))
        fig.suptitle(model + ' - Unit ' + unit + ' scores', fontsize=14)
        for ii, param in zip([0,1,2], ['qc', 'fs', 'u2']):
            print(model, unit, param)
            modpar = model + "_" + param
            df_results = df.loc[df["unit"]==unit, [param, modpar]].dropna()
            y_true = df_results[param].values
            y_pred = df_results[modpar].values
            if y_pred.size>0:
                axs[ii,0] = plot_regression_results(fig, axs[ii,0], y_true, y_pred, param, unit)
                axs[ii,1], std, mu, mae = plot_histogram_results(axs[ii,1], y_true, y_pred, param, unit)
                df_tmp = pd.DataFrame(data={'param': [param], 'model': [model], 'unit': [unit], 'std': [std], 'mu': [mu], 'mae': [mae]})
                df_scores = df_scores.append(df_tmp)
        plt.tight_layout()
        plt.savefig(path_png, dpi=200)
        

#% Save scores to xlsx
path_score = '../../09-Results/Stage-01/Final/CV_Scores.xlsx'
df_scores.to_excel(path_score)
            

#%%Plot global scores heatmap
from matplotlib.colors import LogNorm
unitlistall = np.append(unitlist, 'all')
for ii, param in zip([0,1,2], ['qc', 'fs', 'u2']):
    path_png = '../../09-Results/Stage-01/Final/Std_Heatmap-' + param + '.png'
    fig, axs = plt.subplots(1, 1, figsize=(6,10))
    fig.suptitle('Std [MPa] - ' + param, fontsize=14)
    df_score_plot = pd.DataFrame(index=unitlistall, columns=modellist)
    for model in modellist:
        unitlistmod = df_scores.loc[(df_scores['param']==param) & (df_scores['model']==model), 'unit'].unique()
        for unit in unitlistmod:
            df_score_plot.loc[unit, model] = df_scores.loc[(df_scores['param']==param) & (df_scores['unit']==unit) & (df_scores['model']==model), 'std'].values[0].astype(float)
            df_score_plot.sort_index(inplace=True)
    # if param == 'qc':
    #     df_score_plot[df_score_plot>41] = 41
    # elif param == 'fs':
    #     df_score_plot[df_score_plot>0.66] = 0.66
    # elif param == 'u2':
    #     df_score_plot[df_score_plot>2.3] = 2.3
    axs = sns.heatmap(df_score_plot.astype(float), cmap='RdYlGn_r', linewidths=0.5, annot=True, cbar=False, norm=LogNorm())
    plt.tight_layout()
    plt.savefig(path_png, dpi=200)

    # Save scores to xlsx
    path_score = '../../09-Results/Stage-01/Final/CV_Scores-Global-' + param + '.xlsx'
    df_score_plot.to_excel(path_score)










