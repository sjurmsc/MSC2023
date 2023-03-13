# -*- coding: utf-8 -*-
"""
Created on Tue May  4 15:18:28 2021

@author: GuS
"""

#%% Import libraries
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages
import os

import GM_Toolbox as GMT 

#%% Define functions
def load_obj(path_obj):
    with open(path_obj, 'rb') as f:
        return pickle.load(f)

def plot_axis(ax, xlabel, xminmax, yminmax):
    ''' Function that prepare figure axes with tickmarks'''
    xmin, xmax , ymin, ymax = xminmax[0], xminmax[1], yminmax[0], yminmax[1]
    ax.minorticks_on()
    ax.tick_params(which='major', length=10, width=1, direction='inout', left=True, right=True)
    ax.tick_params(which='minor', length=5, width=1, direction='out', left=True, right=True)
    ax.tick_params(labelbottom=False,labeltop=True)
    ax.set_xlabel(xlabel,labelpad=10)
    ax.xaxis.set_ticks_position('both')
    ax.xaxis.set_label_position('top')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.invert_yaxis()
    return ax

def plot_ICN_lim(ax, yminmax):
    zmin, zmax = yminmax[0], yminmax[1]
    SBT_val=np.array([1.31,2.05,2.6,2.95,3.6,4])
    SBT_des=['Dense sand to gravelly sand ','Sands: clean sands to silty sands',
         'Sand mixtures: silty sand to sandy silt','Silt mixtures: clayey silt & silty clay',
         'Clays: clay to silty clay','Clay - organic soil']
    SBT0=0
    for SBT_i,SBT_n in zip(SBT_val,SBT_des):
        ax.plot([SBT_i,SBT_i],[zmin,zmax],'-',lw=0.5,c='0.55') 
        ax.text(SBT0+(SBT_i-SBT0)/2+0.05, zmax-1, SBT_n, fontsize=6,weight='normal',rotation=90, rotation_mode='anchor')
        SBT0=SBT_i  
    return ax
        
def plot_unit(ax, xminmax, df_CPT, df_unit_col, unittype='seis'):
    if unittype=='seis':
        U='unit'
    elif unittype=='geo':
        U='unit_geo'
    for unit in df_CPT[U].dropna().unique():
        top, bot = df_CPT.loc[df_CPT[U]==unit, 'z_bsf'].min(), df_CPT.loc[df_CPT[U]==unit, 'z_bsf'].max()
        col=df_unit_col[unit].values
        ax.fill_between(xminmax, top, bot, alpha=0.05, color=col)
        ax.fill_between(xminmax, top, bot, alpha=0.05, color=col)
        ax.plot(xminmax, [top, top], '--', linewidth=0.5, color=col)
        if unit == 'GGB01A':
            unit = 'GB01A'
        elif unit == 'GGB01B':
            unit = 'GB01B'
        ax.text(xminmax[1], top, unit,horizontalalignment='right',verticalalignment='top', color=col)
        
colmax=np.array([[142,70,149], [0,152,129],[100, 100,100],[240,64,40],  
                 [34,181,233],[103,143,102],[202,0,197],[255,230,25]])/256
colmin=np.array([[229,203,228],[182,225,194],[200,200,200],[250,167,148],
                 [179,227,238],[191,194,107],[230,185,205],[255,255,174]])/256

def color_range(N,cmax,cmin,unitcount):
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(cmax[0],cmin[0], N)
    vals[:, 1] = np.linspace(cmax[1],cmin[1], N)
    vals[:, 2] = np.linspace(cmax[2],cmin[2], N)
    vals[:,0:3]=vals[:,0:3]
    vals=np.round(vals,3)
    return vals[unitcount,:]



def attri_color(unitlist, colmax, colmin):
    classunit={}
    ii=0
    for unit in unitlist:
        if unit[-1].isalpha():
            if unit[:-2] not in classunit:
                classunit[unit[:-2]]=[ii]
                ii+=1
            classunit[unit[:-2]].append(unit[-2:])
        else:
            if unit[:-1] not in classunit:
                classunit[unit[:-1]]=[ii]
                ii+=1
            classunit[unit[:-1]].append(unit[-1])
            
    
    df_unit_col = pd.DataFrame()
    unitcount={}
    for unit in unitlist:
        if unit[-1].isalpha():
            N=len(classunit[unit[:-2]])-1
            if unit[:-2] not in unitcount:
                unitcount[unit[:-2]]=0   
            else:
                unitcount[unit[:-2]]+=1
            main_unit_ind=classunit[unit[:-2]][0]
            ind_sub_unit=unitcount[unit[:-2]]
        else:
            N=len(classunit[unit[:-1]])-1
            if unit[:-1] not in unitcount:
                unitcount[unit[:-1]]=0   
            else:
                unitcount[unit[:-1]]+=1
            main_unit_ind=classunit[unit[:-1]][0]
            ind_sub_unit=unitcount[unit[:-1]]
            
        cmp=color_range(N,colmax[main_unit_ind], colmin[main_unit_ind], ind_sub_unit)
        df_unit_col[unit] = cmp
    return df_unit_col     


def plot_data_fit_RF(ax, X, y, y_pred):
    model_dist, mae, accuracy, mu, std, mape = GMT.evaluate_modeldist_norm(y, y_pred)
    # model_dist, mae, accuracy, mu, std = GMT.evaluate_modeldist_norm(y, y_pred)
    ax.plot(y_pred, X, 'k', linestyle='--', linewidth=0.75)
    # X_std = np.array([X.min(), X.max()])
    # y_std = np.array([y_pred.min(), y_pred.max()])
    ax.fill_betweenx(X, y_pred*(1-std), y_pred*(1+std), alpha=0.2, color='k', edgecolor='k')
    return ax

def plot_data_fit_mev(ax, X, y_min, y_max):
    ax.fill_betweenx(X, y_min, y_max, alpha=0.2, color='k', edgecolor='k')
    return ax

def plot_data_fit_mev_ann(ax, X, y_min, y_max, y_be):
    ax.fill_betweenx(X, y_min, y_max, alpha=0.2, color='k', edgecolor='k')
    ax.plot(y_be, X, 'k:', linewidth=0.75)
    return ax


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


def plot_data_pred(df, df_mev, df_unit_col, path_pdf):
    # List of locations and units
    loclist = df['ID'].unique()
    # create a PdfPages object
    # pdf = PdfPages(path_pdf)
    # Plot Random Forest classification from Mark
    # Loop over locations
    # for loc in loclist:
    for loc in [5]:
        if loc <= 84:
            loc_txt = 'TNW%03d' %loc 
        else:
            loc_txt = 'TNWTT%02d' %(loc-84)
        print('Loc:', int(loc))
        # path png
        path_png = '../../09-Results/Stage-01/Final/Plots_AiANN/AIANN-Location_%s.png' %(loc_txt)
        # get all CPT at this location
        df_loc = df.loc[df['ID']==loc,:]
        # setup figure and axs
        fig, axs = plt.subplots(1, 5, sharey=True, figsize=(16,7), gridspec_kw={'width_ratios': [1, 1, 1, 1, 4/6]})
        # Loop over CPT at location
        for CPT in df_loc['borehole'].unique():
            print('\tCPT:', CPT)
            df_CPT = df_loc.loc[(df_loc['borehole']==CPT) & (df_loc['z_bsf']>=-1), :]
            z = df_CPT['z_bsf']
            qc = df_CPT['qc']
            fs = df_CPT['fs']
            u2 = df_CPT['u2']
            ICN = df_CPT['ICN']
            # Plot data
            axs[0].invert_yaxis()
            axs[0].plot(qc, z, '.', markersize=1, label=CPT)
            axs[1].plot(fs, z, '.', markersize=1)
            axs[2].plot(u2, z, '.', markersize=1)       
            axs[3].plot(ICN, z, '.', markersize=1)
            
        # AI+ANN from Mark
        for ii, feature in zip([0, 1, 2],['qc', 'fs', 'u2']):
            df_data = df_mev_ann.loc[df_mev_ann['ID']==loc, [feature+'_be', feature+'_min', feature+'_max', 'z_bsf']].dropna().sort_values(by='z_bsf')
            print(df_data)
            X_pred = df_data.loc[:, 'z_bsf'].values
            y_be = df_data.loc[:, feature+'_be'].values
            y_min = df_data.loc[:, feature+'_min'].values
            y_max = df_data.loc[:, feature+'_max'].values
            axs[ii] = plot_data_fit_mev_ann(axs[ii], X_pred, y_min, y_max, y_be)
                
        # Plot SBT
        plot_ICN_lim(axs[3], yminmax)        
        # Plot seis vs geotech units
        plot_unit(axs[4], [0,1], df_CPT, df_unit_col, unittype='seis')
        plot_unit(axs[4], [1,2], df_CPT, df_unit_col, unittype='geo')    
        axs[4].plot([1,1],[0,80],color=[0.6,0.6,0.6],linewidth=1)
        axs[4]=plot_axis(axs[4],'',[0,2], yminmax)
        axs[4].axes.xaxis.set_ticklabels([])
        axs[4].set_title('Units', pad=28,fontsize=11)
        axs[4].text(0.35,-3.5,'Seis',fontsize=11)
        axs[4].text(1.2,-3.5,'Geotech',fontsize=11)            
        # Plot units
        for i in np.arange(0,4):
            plot_unit(axs[i], df_xminmax.iloc[:,i].values, df_CPT, df_unit_col, unittype='geo')                
        # Plot axis
        axs[0].set_ylabel('Depth below seafloor (m)')
        axs[0]=plot_axis(axs[0],'$q_c$ (MPa)', df_xminmax.iloc[:,0].values, yminmax)
        axs[1]=plot_axis(axs[1],'$f_s$ (MPa)', df_xminmax.iloc[:,1].values, yminmax)
        axs[2]=plot_axis(axs[2],'$u_2$ (MPa)', df_xminmax.iloc[:,2].values, yminmax)
        axs[3]=plot_axis(axs[3],'$ICN$ (-)', df_xminmax.iloc[:,3].values, yminmax)
        axs[0].xaxis.set_minor_locator(MultipleLocator(5))
        axs[0].set_ylim(z_max ,z_min)
        axs[0].legend(loc=4, markerscale=7)
        legend_elements = [Line2D([0], [0], color='g', ls=':', lw=.75, label='AI+ANN')]
        axs[1].legend(handles=legend_elements, loc=4)  
    
        # Save to PNG
        # plt.savefig(path_png, dpi=200) 
        
        # Save to PDF    
    #     pdf.savefig(fig)        
    # pdf.close()    
    return axs, fig
    



#%% Load database
path_database = '../../09-Results/Stage-01/Database.pkl'
df = pd.read_pickle(path_database)
# Limit depth
z_min, z_max = -1, 80
# z_min, z_max = df['z_bsf'].min(), df['z_bsf'].max()
df = df[(df['z_bsf']>=z_min) & (df['z_bsf']<=z_max)]


#%% Load Rf classificationb from Mark
# path_dir_pred = '../../09-Results/Stage-01/FromMark/RF_Classification/Ranges'
# df_mev = load_RF_mev(path_dir_pred)

#%% Load AI-ANN MEV
path_dir_ann = '../../09-Results/Stage-01/FromMark/AI_ANN/Ranges'
df_mev_ann = load_RF_mev_ann(path_dir_ann)

#%% Main loop over CPT location (ID)
# Define xmin/xmax for each plot
qcmin, qcmax = 0, 100
fsmin, fsmax = -1, 5
u2min, u2max = -1, 5
Icmin, Icmax = 0, 4
df_xminmax = pd.DataFrame({'qc':[qcmin, qcmax], 'fs':[fsmin, fsmax],
                            'u2':[u2min, u2max], 'ICN':[Icmin, Icmax]})
yminmax = [0, 80]

# List of locations and units
loclist = df['ID'].unique()
unitlist = df['unit'].dropna().unique()    
# Define color for units
unitlist_col=pd.DataFrame(unitlist).sort_values(0).values.flatten()
df_unit_col=attri_color(unitlist_col,colmax,colmin)

# Path to store PDF
path_pdf = '../../09-Results/Stage-01/RandomForest-Results.pdf'
axs, fig = plot_data_pred(df, df_mev_ann, df_unit_col, path_pdf)









