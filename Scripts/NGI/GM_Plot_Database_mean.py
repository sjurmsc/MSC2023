# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 14:09:22 2021

Plot data base per CPT location

@author: GuS

modif by JRD
- Plot unit updated
take in consideration unit_geo
- Creation of functions color_range and attrib color
Create the color for the plot, one color per main unit and variation for the subunits
Based on cmax and cmin
- Creation of plot_unit_color
Display all the units with associated color
- Change the definition of the color units
- Adding on axis with both the stratigraphy (seis and geotech)
- Arrange some figure organisation

"""

#%% Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages


#%% Define functions
def plot_axis(ax,xlabel,xminmax):
    ''' Function that prepare figure axes with tickmarks'''
    xmin, xmax = xminmax[0], xminmax[1]
    ax.minorticks_on()
    ax.tick_params(which='major', length=10, width=1, direction='inout', left=True, right=True)
    ax.tick_params(which='minor', length=5, width=1, direction='out', left=True, right=True)
    ax.tick_params(labelbottom=False,labeltop=True)
    ax.set_xlabel(xlabel,labelpad=10)
    ax.xaxis.set_ticks_position('both')
    ax.xaxis.set_label_position('top')
    ax.set_xlim([xmin,xmax])
    return(ax)

def plot_ICN_lim(ax, z_min, z_max):
    SBT_val=np.array([1.31,2.05,2.6,2.95,3.6,4])
    SBT_des=['Dense sand to gravelly sand ','Sands: clean sands to silty sands',
         'Sand mixtures: silty sand to sandy silt','Silt mixtures: clayey silt & silty clay',
         'Clays: clay to silty clay','Clay - organic soil']
    SBT0=0
    for SBT_i,SBT_n in zip(SBT_val,SBT_des):
        ax.plot([SBT_i,SBT_i],[z_min,z_max],'-',lw=0.5,c='0.55') 
        ax.text(SBT0+(SBT_i-SBT0)/2+0.05, z_max-1, SBT_n, fontsize=6,weight='normal',rotation=90, rotation_mode='anchor')
        SBT0=SBT_i  
        
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

#%% Load database
path_database = '../../09-Results/Stage-01/Database.pkl'
df = pd.read_pickle(path_database)
path_database = '../../09-Results/Stage-01/Database_mean.pkl'
df_m = pd.read_pickle(path_database)


z_min, z_max = -1, 80
# z_min, z_max = df['z_bsf'].min(), df['z_bsf'].max()
df = df[(df['z_bsf']>=z_min) & (df['z_bsf']<=z_max)]


#%% Main loop over CPT location (ID)
# Define xmin/xmax for each plot
qcmin, qcmax = 0, 100
fsmin, fsmax = -1, 5
u2min, u2max = -1, 5
Icmin, Icmax = 0, 4
Qpmin, Qpmax = 0, 155
Vpmin, Vpmax = 1500, 2150
# xminmax = np.array([[qcmin, qcmax], [fsmin, fsmax], [u2min, u2max],
#                     [Icmin, Icmax], [Qpmin, Qpmax], [Vpmin, Vpmax]])
df_xminxmax = pd.DataFrame({'qc':[qcmin, qcmax], 'fs':[fsmin, fsmax], 'u2':[u2min, u2max],
                    'ICN':[Icmin, Icmax], 'Qp':[Qpmin, Qpmax], 'Vp':[Vpmin, Vpmax],
                    'amp_tr_1':[-25, 25] ,'envelop':[-0.5, 25],'energy':[-0.09, 1]})

# List of attributes to plot
attr_list = ['Qp','Vp','amp_tr_1','envelop','energy']

# List of locations and units
loclist = df['ID'].sort_values().unique()
unitlist = df['unit_geo'].dropna().unique()

# Define color for units
unitlist=pd.DataFrame(unitlist).sort_values(0).values.flatten()
df_unit_col=attri_color(unitlist,colmax,colmin)

# create a PdfPages object
pdf = PdfPages('../../09-Results/Stage-01/Database-Mean.pdf')

# Loop over locations
for loc in loclist:
    print('Loc: ', int(loc))
    # path png
    path_png = '../../09-Results/Stage-01/Final/Plots_Database/Database-Mean-Location_%03d.png' %(loc)
    # get all CPT at this location
    df_loc = df.loc[df['ID']==loc,:]
    df_loc_m = df_m.loc[df_m['ID']==loc,:]
    # setup figure and axs based on database
    n_attr = len(attr_list)
    fig, axs = plt.subplots(1, 5+n_attr, sharey=True,figsize=(31,7))
    # Loop over CPT at location
    for CPT in df_loc['borehole'].unique():
        print(CPT)
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
    # Plot Mean
    z_m = df_loc_m['z_bsf']
    qc_m = df_loc_m['qc']
    fs_m = df_loc_m['fs']
    u2_m = df_loc_m['u2']
    ICN_m = df_loc_m['ICN']
    axs[0].plot(qc_m, z_m, 'k.', markersize=.5, label='mean')
    axs[1].plot(fs_m, z_m, 'k.', markersize=.5)
    axs[2].plot(u2_m, z_m, 'k.', markersize=.5)       
    axs[3].plot(ICN_m, z_m, 'k.', markersize=.5)
        
    # Plot SBT
    plot_ICN_lim(axs[3], z_min, z_max)
        
    # Plot seismic attributes (Qp, Vp, ...)
    if n_attr >= 1:
        ii = 4
        for attr in attr_list:
            print(attr)
            axs[ii].plot(df_CPT[attr], z, markersize=1)
            axs[ii]=plot_axis(axs[ii], attr, df_xminxmax.iloc[:,ii].values)
            ii = ii+1
            
    # Plot units
    for i in np.arange(0,4+n_attr):
        plot_unit(axs[i], df_xminxmax.iloc[:,i].values, df_loc, df_unit_col,unittype='geo')
    
    # Plot seis vs geotech units
    plot_unit(axs[5+n_attr-1], [0,1], df_CPT, df_unit_col, unittype='seis')
    plot_unit(axs[5+n_attr-1], [1,2], df_CPT, df_unit_col, unittype='geo')    
    axs[5+n_attr-1].plot([1,1],[0,80],color=[0.6,0.6,0.6],linewidth=1)

    # Plot axis
    axs[0].set_ylabel('Depth below seafloor (m)')
    axs[0]=plot_axis(axs[0],'$q_c$ (MPa)', df_xminxmax.iloc[:,0].values)
    axs[1]=plot_axis(axs[1],'$f_s$ (MPa)', df_xminxmax.iloc[:,1].values)
    axs[2]=plot_axis(axs[2],'$u_2$ (MPa)', df_xminxmax.iloc[:,2].values)
    axs[3]=plot_axis(axs[3],'$ICN$ (-)', df_xminxmax.iloc[:,3].values)
    axs[5+n_attr-1]=plot_axis(axs[5+n_attr-1],'',[0,2])
    axs[5+n_attr-1].axes.xaxis.set_ticklabels([])
    axs[5+n_attr-1].set_title('Units', pad=28,fontsize=11)
    axs[5+n_attr-1].text(0.35,-3.5,'Seis',fontsize=11)
    axs[5+n_attr-1].text(1.2,-3.5,'Geotech',fontsize=11)
    axs[0].xaxis.set_minor_locator(MultipleLocator(5))
    axs[0].set_ylim(z_max ,z_min)
    axs[0].legend(loc=4,markerscale=7)   
    
    # Add distance to trace text box
    dtxt = 'Distance to closest seismic trace: %.2f m' % np.min(df_loc['dist_tr_1'].unique())
    print(dtxt)
    plt.subplots_adjust(bottom = 0.05)
    plt.gcf().text(0.06, 0.035, dtxt, fontsize=10)
    # Save to PDF    
    plt.subplots_adjust(left=0.06, right=0.95, top=0.9, bottom=0.08)
    # Save to PNG
    plt.savefig(path_png, dpi=200) 
#     #save to pdf
#     pdf.savefig(fig)
pdf.close()  
   


