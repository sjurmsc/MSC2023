# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 10:58:06 2021

Plot:
    CPT data for each location
    Linear fit and Kriged linear fit
    Xplot and histogram of Kriging Cross-validation

@author: GuS
"""


#%% Import libraries
import pickle
import numpy as np
import pandas as pd
import netCDF4 as nc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

from scipy.stats import norm

import GM_Toolbox as GMT 

#%% Define functions
def load_obj(path_obj):
    with open(path_obj, 'rb') as f:
        return pickle.load(f)

def plot_regression_results(ax, y_true, y_pred, feature, unit, scores=[]):
    model_dist, mae, accuracy, mu, std, mape = GMT.evaluate_modeldist_norm(y_true, y_pred)
    if not scores:
        scores_txt = (r'$MAE={:.2f}$' + '\n' + r'$Accuracy={:.2f}$ %').format(mae, accuracy)
    else:
        mae, mae_std = scores[0], scores[1]
        scores_txt = (r'$MAE={:.2f} \pm {:.2f}$' + '\n' + r'$Accuracy={:.2f}$ %').format(mae, mae_std, accuracy)
    """Scatter plot of the predicted vs true targets."""
    ax.plot(y_true, y_pred, '.', alpha=0.2,  color='tab:blue', markersize=5)
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

def plot_CPT_slopinter_results(ax1, x, slope, intercept, Kslope, Kintercept, feature, unit):
    """Scatter plot of the slope or intercept for each location, for a given unit"""
    color = 'tab:blue'
    ax1.plot(x, slope, '.', alpha=1,  color=color, markersize=5)
    ax1.plot(x, Kslope, '+', alpha=1,  color=color, markersize=7, markeredgewidth=0.75)
    ax1.set_xlabel('CPT #')
    ax1.set_ylabel('Predicted slope', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.plot(x, intercept, '.', alpha=1,  color=color, markersize=5)
    ax2.plot(x, Kintercept, '+', alpha=1,  color=color, markersize=7, markeredgewidth=0.75)
    ax2.set_ylabel('Predicted intercept', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    title = unit + ' - Slope/intercept per CPT'
    ax1.set_title(title)
    legend_elements = [Line2D([0], [0], marker='+', color='w', label='Kriged Linear Fit', markeredgecolor='k', markersize=7),
                       Line2D([0], [0], marker='.', color='w', label='Linear Fit',  markerfacecolor='k', markersize=7)]
    ax1.legend(handles=legend_elements, loc=4)
    ax1.grid(True)
    return ax1

def plot_CPT_slopinter_Xplot(ax1, slope, intercept, Kslope, Kintercept, feature, unit):
    color = 'tab:blue'
    ax1.plot([slope.min(), slope.max()], [slope.min(), slope.max()],'--', color=color, linewidth=2)
    ax1.plot(slope, Kslope, '.', alpha=0.5,  color=color, markersize=5)
    ax1.set_xlabel('Measured $%s_{%s}$ slope' %(feature[0], feature[1]), color=color)
    ax1.set_ylabel('Predicted $%s_{%s}$ slope' %(feature[0], feature[1]), color=color)
    ax1.tick_params(axis='x', labelcolor=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    color = 'tab:green'
    ax2 = inset_axes(ax1,
                    height="100%", # set height
                    width="100%",   # and width
                    loc=10) 
    ax2.patch.set_alpha(0)
    ax2.plot([intercept.min(), intercept.max()], [intercept.min(), intercept.max()],'--r', color=color, linewidth=2)
    ax2.plot(intercept, Kintercept, '.', alpha=0.5,  color=color, markersize=5)
    ax2.set_xlabel('Measured $%s_{%s}$ intercept' %(feature[0], feature[1]), color=color)
    ax2.set_ylabel('Predicted $%s_{%s}$ intercept' %(feature[0], feature[1]), color=color)
    ax2.xaxis.tick_top()
    ax2.yaxis.tick_right()
    ax2.xaxis.set_label_position('top') 
    ax2.yaxis.set_label_position('right') 
    ax2.tick_params(axis='x', labelcolor=color)
    ax2.tick_params(axis='y', labelcolor=color)
    title = unit + ' - Slope/intercept Xplot'
    ax1.set_title(title, y=0.0)
    ax1.grid(True)
    return ax1

def plot_pred_grd(ax, za, lreg, xx, yy, CPT_x, CPT_y, unit, txt_lgd):
    lreg = np.array([x for x in lreg]).flatten()
    vmin = np.floor(np.min([np.min(za), np.min(lreg)]))
    vmax = np.ceil(np.max([np.max(za), np.max(lreg)]))
    # ax.set_aspect(1)
    # c = ax.scatter(xyz[:,1], xyz[:,0], c=za, cmap='viridis', vmin=vmin, vmax=vmax)
    c = ax.imshow(za, origin='lower', extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                  cmap='viridis', vmin=vmin, vmax=vmax, aspect=2)
    ax.scatter(CPT_x, CPT_y, c=lreg, cmap='viridis', edgecolors='k', vmin=vmin, vmax=vmax)
    # for index, row in df_CPT_loc.iterrows():
    #     ax.annotate(index, (row['x'], row['y']+2), size=12)
    cbar = plt.colorbar(c, ax=ax)
    cbar.set_label(txt_lgd)
    ax.set_title(txt_lgd + '-' + unit)
    ax.set_xlabel('UTM X [m]')
    ax.set_ylabel('UTM Y [m]')
    ax.grid(color="0.5", linestyle=':', linewidth=0.5)
    return ax

def plot_variogram(ax, kr, unit, txt_lgd):
    ax.plot(kr.lags, kr.semivariance, 'r*')
    ax.plot(kr.lags, 
            kr.variogram_function(kr.variogram_model_parameters, kr.lags), 'k-')
    ax.set_title(txt_lgd + '-' + unit)
    ax.set_xlabel('Distance')
    ax.set_ylabel('Semivariance')
    return ax

#%% Load database
path_database = '../../09-Results/Stage-01/Database.csv'
df = pd.read_csv(path_database)
# Limit depth
z_min, z_max = -1, 80
# z_min, z_max = df['z_bsf'].min(), df['z_bsf'].max()
df = df[(df['z_bsf']>=z_min) & (df['z_bsf']<=z_max)]

#%% Load local linear fit KFold results and Kriged results
path_dict = '../../09-Results/Stage-01/LocalLinearFit-Kfold-Results_bounds.pkl'
df_lreg = load_obj(path_dict)
path_dict = '../../09-Results/Stage-01/LocalLinearFitwBounds-UKriging-GroupKfold-Results.pkl'
df_lregKri = load_obj(path_dict)
path_dict = '../../09-Results/Stage-01/LocalLinearFitwBounds-UKriging-GroupKfold-Scores.pkl'
df_lregKriSco = load_obj(path_dict)

#%% Load Seafloor
filepath = "P:/2019/07/20190798/Calculations/Bathymetry/MMT_RVO_TNW_EM2040D_5m_alldata.grd"
fh = nc.Dataset(filepath, mode='r')
xx = fh.variables['x'][:]
yy = fh.variables['y'][:]
zz = fh.variables['z'][:]
mask = np.ma.getmask(zz)
fh.close()
# xx, yy = np.meshgrid(np.float32(x), np.float32(y))
# xyz = np.c_[xx.ravel(), yy.ravel(), z.ravel()]
# xyz = xyz[~np.isnan(xyz).any(axis=1)]

#%% Main loop over CPT location (ID)
# create a PdfPages object
pdf = PdfPages('../../09-Results/Stage-01/LocalLinearFit-UKriging_Spherical18-Scores.pdf')

# Loop over units
unitlist = df['unit'].dropna().unique()
for unit in unitlist:
    print('unit: ' + unit)
    # setup figure and axs
    fig, axs = plt.subplots(3, 5, figsize=(35,14))
    fig.suptitle(unit + ' - Kriged Linear Regression metrics', fontsize=14)
    for ii, feature in zip([0,1,2], ['qc', 'fs', 'u2']):
        df_data = df.loc[df['unit']==unit,['z_bsf', 'ID', feature]].dropna()
        loclist = df_data['ID'].unique()
        # Loop over locations
        df_Y_pred = pd.DataFrame([])
        for loc in loclist:
            df_y_pred = pd.DataFrame([])
            df_loc = df_data.loc[df_data['ID']==loc,:]
            X = df_loc['z_bsf']            
            y = df_loc[feature]
            slope = df_lregKri.loc[(df_lregKri['feature']==feature) & (df_lregKri['ID']==loc) & (df_lregKri['unit']==unit),'slope'].values
            intercept = df_lregKri.loc[(df_lregKri['feature']==feature) & (df_lregKri['ID']==loc) & (df_lregKri['unit']==unit),'intercept'].values
            if slope.size>0:
                y_pred = slope*X + intercept*np.ones(np.shape(X))
                df_y_pred['y'] = y.values
                df_y_pred['y_pred'] = y_pred.values
                df_y_pred['unit'] = unit
                df_y_pred['feature'] = feature
                df_y_pred['ID'] = loc                
                df_Y_pred = df_Y_pred.append(df_y_pred, ignore_index=True)
                
        if unit in df_lregKriSco['unit'].unique():
            [r2] = df_lregKriSco.loc[(df_lregKriSco['feature']==feature) & (df_lregKriSco['unit']==unit),'slope_r2']
            [r2_std] = df_lregKriSco.loc[(df_lregKriSco['feature']==feature) & (df_lregKriSco['unit']==unit),'slope_r2_std']
            [mae] = df_lregKriSco.loc[(df_lregKriSco['feature']==feature) & (df_lregKriSco['unit']==unit),'slope_mae']
            [mae_std] = df_lregKriSco.loc[(df_lregKriSco['feature']==feature) & (df_lregKriSco['unit']==unit),'slope_mae_std']
            scores_txt = (r'$R^2={:.2e} \pm {:.2e}$' + '\n' + r'$MAE={:.2e} \pm {:.2e}$').format(r2, r2_std, mae, mae_std)     
            scores = [mae, mae_std]
            CPT_x = df_lreg.loc[(df_lreg['feature']==feature) & (df_lreg['unit_geo']==unit),'x'].values
            CPT_y = df_lreg.loc[(df_lreg['feature']==feature) & (df_lreg['unit_geo']==unit),'y'].values
            slopes = df_lreg.loc[(df_lreg['feature']==feature) & (df_lreg['unit_geo']==unit),'slope'].values
            intercepts =  df_lreg.loc[(df_lreg['feature']==feature) & (df_lreg['unit_geo']==unit),'intercept'].values
            Kslopes = df_lregKri.loc[(df_lregKri['feature']==feature) & (df_lregKri['unit']==unit),'slope'].values
            Kintercepts =  df_lregKri.loc[(df_lregKri['feature']==feature) & (df_lregKri['unit']==unit),'intercept'].values
            [K_slope] = df_lregKriSco.loc[(df_lregKriSco['feature']==feature) & (df_lregKriSco['unit']==unit),'slope_kri_obj']
            if isinstance(K_slope, float):
                za_slope = np.ma.array(K_slope*np.ones(np.shape(np.meshgrid(xx,yy))[-2:]), mask=mask)
                ssa_slope = np.ma.array(np.ones(np.shape(np.meshgrid(xx,yy))[-2:]), mask=mask)
            else:
                za_slope, ssa_slope = K_slope.execute('masked', xx, yy, mask=mask)
            [K_intercept] = df_lregKriSco.loc[(df_lregKriSco['feature']==feature) & (df_lregKriSco['unit']==unit),'intercept_kri_obj']
            if isinstance(K_intercept, float):
                za_intercept = np.ma.array(K_intercept*np.ones(np.shape(np.meshgrid(xx,yy))[-2:]), mask=mask)
                ssa_intercept = np.ma.array(np.ones(np.shape(np.meshgrid(xx,yy))[-2:]), mask=mask)
            else:
                za_intercept, ssa_intercept = K_intercept.execute('masked', xx, yy, mask=mask)
            
            # Plot Kriging KFold crossvalidation cross plot and histogram
            axs[ii,0] = plot_regression_results(axs[ii,0], df_Y_pred['y'].values, df_Y_pred['y_pred'].values, feature, unit, scores=scores)
            axs[ii,1] = plot_histogram_results(axs[ii,1], df_Y_pred['y'].values, df_Y_pred['y_pred'].values, feature, unit)  
            
            # Plot comparison of slope and intercept from KFold crossval kriging and linear regression
            # axs[ii,2] = plot_CPT_slopinter_results(axs[ii,2], loclist, slopes, intercepts, Kslopes, Kintercepts, feature, unit)
            axs[ii,2] = plot_CPT_slopinter_Xplot(axs[ii,2], slopes, intercepts, Kslopes, Kintercepts, feature, unit)
            # axs[ii,3] = plot_regression_results(axs[ii,3], slopes, Kslopes, scores_txt, unit, feature)
            # axs[ii,4] = plot_regression_results(axs[ii,4], intercepts, Kintercepts, scores_txt, unit, feature)
            
            # # Plot predicted slope and intercept grids & kriging semi-variograms 
            if not isinstance(K_slope, float):
                axs[ii,3] = plot_pred_grd(axs[ii,3], za_slope, slopes, xx, yy, CPT_x, CPT_y, unit, 'slope')
                axs[ii,4] = plot_variogram(axs[ii,4], K_slope, unit, 'slope')
            # else:
            #     axs[ii,5].axis('off')
            if not isinstance(K_intercept, float):
                axs[ii,3] = plot_pred_grd(axs[ii,3], za_intercept, intercepts, xx, yy, CPT_x, CPT_y, unit, 'intercept')
                axs[ii,4] = plot_variogram(axs[ii,4], K_intercept, unit, 'intercept')
            # else:
            #     axs[ii,5].axis('off')

    plt.tight_layout()
    # Save to PDF    
    pdf.savefig(fig)
pdf.close()  




