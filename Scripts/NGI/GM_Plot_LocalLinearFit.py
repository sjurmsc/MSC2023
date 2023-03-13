# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 13:57:23 2021

Plot CPT from all location and linear fit per unit

@author: GuS
"""

#%% Import libraries
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator

from scipy.stats import norm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.base import BaseEstimator 
from sklearn.base import RegressorMixin

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
    
def plot_unit(ax, xminmax, df_CPT, df_unit_col, unittype='geo'):
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

class RTK_Regressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        # The prediction of simple model 
        if Ulvl!='0': # fit also the intercept
            lin_Hub = linear_model.HuberRegressor(epsilon=1.35)
        else:  # if the unit is 0, then itnercept is set to 0.
            lin_Hub = linear_model.HuberRegressor(epsilon=1.35,fit_intercept=False)   
            
        # Filtering to avoid to big influence of sand lines
        X0,y0=data_preparation(X,y,0,detrend=0,bounds=[-1, 1])
        model=lin_Hub.fit(X0,y0)
        y_pred=model.predict(X)
        
        X1,y1,_=data_preparation(X,y,y_pred,detrend='detrend', bounds=[-1,1])
#        self.params_ = np.polyfit(z0, y0,1)[:,0]
        self.model  = lin_Hub.fit(X1,y1)
        self.coef_   = lin_Hub.coef_
        self.intercept_   = lin_Hub.intercept_
        # self.coef_   = model.coef_
        # self.intercept_   = model.intercept_
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
def plot_data_fit_RTK(ax, X, lin_reg, feature, xminmax, yminmax):
    ax.plot(lin_reg.predict(X), X, 'k', linewidth=0.5)
    return ax

def plot_data_fit(ax, X, slope, inter):
    ax.plot(linear_fit(X,slope,inter), X, 'k', linewidth=0.5)
    return ax

def plot_regression_results(ax, y_true, y_pred, scores, unit, df_col):
    """Scatter plot of the predicted vs true targets."""
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],'--r', linewidth=2)
    ax.plot(y_true, y_pred, '.', alpha=0.2,  color=df_col[unit], markersize=5)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xlim([y_true.min(), y_true.max()])
    ax.set_ylim([y_true.min(), y_true.max()])
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                          edgecolor='none', linewidth=0)
    ax.legend([extra], [scores], loc='upper left', prop={'size': 10})
    title = unit + ' - Cross-validation'
    ax.set_title(title)
    ax.grid(True)
    return ax
    
def plot_histogram_results(ax, y_true, y_pred, feature, unit):
    model_dist=(y_pred[y_true>0]-y_true[y_true>0])/y_true[y_true>0]
    # model_dist=(y_pred-y_true)
    # best fit of data
    (mu, std) = norm.fit(model_dist)
    # the histogram of the data
    ax.hist(model_dist, 60, edgecolor='black', density=True, facecolor='green', alpha=0.5)
    # add a 'best fit' line
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2)
    #plot
    ax.set_xlabel('$(%s_{%s_{pred}}-%s_{%s})/%s_{%s}$'%(feature[0], feature[1],feature[0], feature[1],feature[0], feature[1]))
    # ax.set_xlabel('$%s_{%s_{pred}}-%s_{%s}$'%(feature[0], feature[1], feature[0], feature[1]))
    ax.set_ylabel('Probability')
    ax.set_title(unit + ' - Histogram')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                          edgecolor='none', linewidth=0)
    ax.legend([extra], ['$\mu=%.3f$\n$\sigma=%.3f$' %(mu, std)], loc='upper left', prop={'size': 10})
    ax.grid(True)
    return ax

# def plot_LocalLinFit_old(df, df_lreg, path_pdf, df_xminxmax):
#     # Limit depth
#     z_min, z_max = -1, 80
#     # z_min, z_max = df['z_bsf'].min(), df['z_bsf'].max()
#     df = df[(df['z_bsf']>=z_min) & (df['z_bsf']<=z_max)]

#     yminmax = [0, 80]
#     # List of locations and units
#     loclist = df['ID'].unique()
#     unitlist = df['unit'].dropna().unique()    
#     # Define color for units
#     unitlist_col=pd.DataFrame(unitlist).sort_values(0).values.flatten()
#     df_unit_col=attri_color(unitlist_col,colmax,colmin)    
#     # create a PdfPages object
#     # pdf = PdfPages(path_pdf)
#     # Loop over locations
#     for loc in loclist[0:3]:
#         print('Loc:', int(loc))
#         # get all CPT at this location
#         df_loc = df.loc[df['ID']==loc,:]
#         # setup figure and axs
#         fig, axs = plt.subplots(1, 4, sharey=True,figsize=(14,7))
#         # Loop over CPT at location
#         for CPT in df_loc['borehole'].unique():
#             print('\tCPT:', CPT)
#             df_CPT = df_loc.loc[(df_loc['borehole']==CPT) & (df_loc['z_bsf']>=-1), :]
#             z = df_CPT['z_bsf']
#             qc = df_CPT['qc']
#             fs = df_CPT['fs']
#             u2 = df_CPT['u2']
#             ICN = df_CPT['ICN']
#             # Plot data
#             axs[0].invert_yaxis()
#             axs[0].plot(qc, z, '.', markersize=1, label=CPT)
#             axs[1].plot(fs, z, '.', markersize=1)
#             axs[2].plot(u2, z, '.', markersize=1)       
#             axs[3].plot(ICN, z, '.', markersize=1)
#         # Plot linear fit
#         for ii, feature in zip([0, 1, 2],['qc', 'fs', 'u2']):
#             df_loc = df.loc[df['ID']==loc,:]
#             # Loop over units
#             for unit in df_loc['unit_geo'].dropna().unique():
#                 df_data = df_loc.loc[df_loc['unit_geo']==unit,['z_bsf', feature]].dropna()
#                 if not df_data.empty:
#                     X = df_data['z_bsf'].values.reshape(-1,1)
#                     y = df_data[feature].values.reshape(-1,1)
#                     [lin_reg] = df_lreg.loc[(df_lreg['feature']==feature) & (df_lreg['ID']==loc) & (df_lreg['unit_geo']==unit),'lreg']
                    
#                     X_pred = df_loc.loc[df_loc['unit_geo']==unit,['z_bsf']].drop_duplicates()
#                     axs[ii] = plot_data_fit_RTK(axs[ii], X_pred, lin_reg, 
#                                             feature, df_xminmax.loc[:,feature].values, yminmax)            
#         # Plot SBT
#         plot_ICN_lim(axs[3], yminmax)            
#         # Plot units
#         for i in np.arange(0,4):
#             plot_unit(axs[i], df_xminmax.iloc[:,i].values, df_CPT, df_unit_col, unittype='geo')    
#         # Plot axis
#         axs[0].set_ylabel('Depth below seafloor (m)')
#         axs[0]=plot_axis(axs[0],'$q_c$ (MPa)', df_xminmax.iloc[:,0].values, yminmax)
#         axs[1]=plot_axis(axs[1],'$f_s$ (MPa)', df_xminmax.iloc[:,1].values, yminmax)
#         axs[2]=plot_axis(axs[2],'$u_2$ (MPa)', df_xminmax.iloc[:,2].values, yminmax)
#         axs[3]=plot_axis(axs[3],'$ICN$ (-)', df_xminmax.iloc[:,3].values, yminmax)
#         axs[0].xaxis.set_minor_locator(MultipleLocator(5))
#         axs[0].set_ylim(z_max ,z_min)
#         axs[0].legend(loc=4,markerscale=7)  
#         # Save to PDF    
#         pdf.savefig(fig)        
#     pdf.close()  

def linear_fit(x,slope,inter):
    return slope*x + inter

def plot_LocalLinFit(df, df_lreg, path_pdf, df_xminxmax,pdfpng='pdf'):
    # Limit depth
    z_min, z_max = -1, 80
    # z_min, z_max = df['z_bsf'].min(), df['z_bsf'].max()
    df = df[(df['z_bsf']>=z_min) & (df['z_bsf']<=z_max)]

    yminmax = [0, 80]
    # List of locations and units
    loclist = df['ID'].unique()
    unitlist = df['unit_geo'].dropna().unique()    
    # Define color for units
    unitlist_col=pd.DataFrame(unitlist).sort_values(0).values.flatten()
    df_unit_col=attri_color(unitlist_col,colmax,colmin)    
    
    # create a PdfPages object
    if pdfpng=='pdf':
        pdf = PdfPages(path_pdf)
    # Loop over locations
    for loc in loclist:
        print('Loc:', int(loc))
        # get all CPT at this location
        df_loc = df.loc[df['ID']==loc,:]
        # setup figure and axs
        fig, axs = plt.subplots(1, 4, sharey=True,figsize=(14,7))
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
 
        # Plot linear fit
        for ii, feature in zip([0, 1, 2],['qc', 'fs', 'u2']):
            df_loc = df.loc[df['ID']==loc,:]
            # Loop over units
            for unit in df_loc['unit_geo'].dropna().unique():
                df_data = df_loc.loc[df_loc['unit_geo']==unit,['z_bsf', feature]].dropna()
                if len(df_data)>10:
                    X = df_data['z_bsf'].values.reshape(-1,1)
                    y = df_data[feature].values.reshape(-1,1)
                    
                    slope=df_lreg.loc[(df_lreg['feature']==feature) & (df_lreg['ID']==loc) & (df_lreg['unit_geo']==unit),'slope'].values[0]
                    intercept=df_lreg.loc[(df_lreg['feature']==feature) & (df_lreg['ID']==loc) & (df_lreg['unit_geo']==unit),'intercept'].values[0]
    
                    axs[ii] = plot_data_fit(axs[ii], X, slope, intercept) 
                            
        # Plot SBT
        plot_ICN_lim(axs[3], yminmax)            
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
        axs[0].legend(loc=4,markerscale=7)  
        # Save to PDF    
        if pdfpng=='png':
            if len(str(int(loc)))==1:
                fname=path_pdf+'loc0'+str(int(loc))+'.png'
            else:
                fname=path_pdf+'loc'+str(int(loc))+'.png'
            plt.savefig(fname,dpi=200)
        if pdfpng=='pdf':
            pdf.savefig(fig)        
    if pdfpng=='pdf':
        pdf.close()  

    
def plot_LocalLinFit_Metrics(df, df_lreg, path_pdf, df_xminmax):
    # Limit depth
    z_min, z_max = -1, 80
    # z_min, z_max = df['z_bsf'].min(), df['z_bsf'].max()
    df = df[(df['z_bsf']>=z_min) & (df['z_bsf']<=z_max)]

    yminmax = [0, 80]    
    # List of locations
    loclist = df['ID'].unique()
    unitlist = df['unit'].dropna().unique()    
    # Define color for units
    unitlist_col=pd.DataFrame(unitlist).sort_values(0).values.flatten()
    df_unit_col=attri_color(unitlist_col,colmax,colmin)  
    # create a PdfPages object
    pdf = PdfPages(path_pdf)
    
    # Loop over locations
    for loc in loclist:
        print('Loc:', int(loc))
        # get all CPT at this location
        df_loc = df.loc[df['ID']==loc,:]
        unitlist = df_loc['unit_geo'].dropna().unique()
        for feature in ['qc', 'fs', 'u2']:
            # setup figure and axs
            fig = plt.figure(figsize=(26,10))
            fig.suptitle('TNW%03d - Linear regression KFold Cross-validation for $%s_{%s}$' %(int(loc), feature[0], feature[1]),
                         fontsize=14)
            gs = GridSpec(3, 9) #, width_ratios=[2,3])
            xx, yy = np.meshgrid([0,1,2],[1,3,5,7])
            xyX = np.column_stack((xx.ravel(), yy.ravel()))
            xx, yy = np.meshgrid([0,1,2],[2,4,6,8])
            xyH = np.column_stack((xx.ravel(), yy.ravel()))
            axD = fig.add_subplot(gs[:,0])        
            # Loop over CPT at location - plot data
            for CPT in df_loc['borehole'].unique():
                print('\tCPT:', CPT)
                df_CPT = df_loc.loc[(df_loc['borehole']==CPT) & (df_loc['z_bsf']>=-1), :]
                z = df_CPT['z_bsf']
                data = df_CPT[feature]
                # Plot feature
                axD.invert_yaxis()
                axD.plot(data, z, '.', markersize=1, label=CPT)    
            # Loop over units - plot linear fit
            ii = 0
            for unit in unitlist:
                df_data = df_loc.loc[df_loc['unit_geo']==unit,['z_bsf', feature]].dropna()
                if not df_data.empty:
                    X = df_data['z_bsf'].values.reshape(-1,1)
                    y = df_data[feature].values.reshape(-1,1)
                    [lin_reg] = df_lreg.loc[(df_lreg['feature']==feature) & (df_lreg['ID']==loc) & (df_lreg['unit']==unit),'lreg']
                    X_pred = df_loc.loc[df_loc['unit_geo']==unit,['z_bsf']].drop_duplicates()
                    axD = plot_data_fit_RTK(axD, X_pred, lin_reg, feature, df_xminmax.loc[:,feature].values, yminmax)
                    n_splits = np.min([5, X.shape[0]])
                    if n_splits > 2:
                        cv = KFold(n_splits=n_splits)
                        y_pred = cross_val_predict(lin_reg, X, y, cv=cv, n_jobs=-1, verbose=0)
                        [r2] = df_lreg.loc[(df_lreg['feature']==feature) & (df_lreg['ID']==loc) & (df_lreg['unit']==unit),'r2']
                        [r2_std] = df_lreg.loc[(df_lreg['feature']==feature) & (df_lreg['ID']==loc) & (df_lreg['unit']==unit),'r2_std']
                        [mae] = df_lreg.loc[(df_lreg['feature']==feature) & (df_lreg['ID']==loc) & (df_lreg['unit']==unit),'mae']
                        [mae_std] = df_lreg.loc[(df_lreg['feature']==feature) & (df_lreg['ID']==loc) & (df_lreg['unit']==unit),'mae_std']
                        scores_txt = (r'$R^2={:.2f} \pm {:.2f}$' + '\n' + r'$MAE={:.2f} \pm {:.2f}$').format(r2, r2_std, mae, mae_std)
                        axX = fig.add_subplot(gs[xyX[ii,0],xyX[ii,1]])
                        axH = fig.add_subplot(gs[xyH[ii,0],xyH[ii,1]])
                        axX = plot_regression_results(axX, y, y_pred, scores_txt, unit, df_unit_col)
                        axH = plot_histogram_results(axH, y, y_pred, feature, unit)
                        ii = ii+1            
            # Plot units
            plot_unit(axD, df_xminmax.loc[:,feature].values, df_CPT, df_unit_col, unittype='geo')                  
            # Plot axis
            axD.set_ylabel('Depth below seafloor (m)')
            axD=plot_axis(axD,'$%s_{%s}$ (MPa)'%(feature[0], feature[1]) , df_xminmax.loc[:,feature].values, yminmax)
            axD.xaxis.set_minor_locator(MultipleLocator(5))
            axD.set_ylim(z_max ,z_min)
            axD.legend(loc=4,markerscale=7)             
            plt.tight_layout()          
            # Save to PDF    
            pdf.savefig(fig)        
    pdf.close()  

def plot_fit_spread(df_lreg,df_GL):
    # List of locations
    loclist = df_lreg['ID'].unique()
    unitlist = df_lreg['unit_geo'].dropna().unique()  
    
    slope_bounds={}
    slope_bounds['qc']=[-1, 6]
    slope_bounds['fs']=[-0.035, 0.05]
    slope_bounds['u2']=[-0.045, 0.09]
    
    inter_bounds={}
    inter_bounds['qc']=[-40,50]
    inter_bounds['fs']=[-0.6,0.85]
    inter_bounds['u2']=[-2.2,2]
    
    
    for unit in sorted(unitlist):
        f,axs=plt.subplots(2,3,figsize=(12,8))
        
        for ii, feature in zip([0, 1, 2],['qc', 'fs', 'u2']):
            df_loc=df_lreg.loc[(df_lreg['unit_geo']==unit) & (df_lreg['feature']==feature)]
            
            sl=[]
            inter=[]
            for loc in loclist:
                df_data=df_loc.loc[(df_loc['ID']==loc)]
                if not df_data.empty:
                    sl.append(df_data['slope'].values[0])
                    inter.append(df_data['intercept'].values[0])
            
            axs[0,ii].plot(sl,'.')
            axs[1,ii].plot(inter,'.')
            
            sl_GL=df_GL.loc[(df_GL['feature']==feature)  & (df_GL['unit_geo']==unit),'slope'].values[0]
            axs[0,ii].plot([0, len(sl)],[sl_GL, sl_GL], alpha=0.8)
            inter_GL=df_GL.loc[(df_GL['feature']==feature)  & (df_GL['unit_geo']==unit),'intercept'].values[0]
            axs[1,ii].plot([0, len(sl)],[inter_GL, inter_GL], alpha=0.8)
            
            axs[0,ii].set_title(feature+' - slope')
            axs[1,ii].set_title(feature+' - intercept')
            
        axs[0,0].set_ylim(slope_bounds['qc'])
        axs[0,1].set_ylim(slope_bounds['fs'])
        axs[0,2].set_ylim(slope_bounds['u2'])
        axs[1,0].set_ylim(inter_bounds['qc'])
        axs[1,1].set_ylim(inter_bounds['fs'])
        axs[1,2].set_ylim(inter_bounds['u2'])
        
        f.suptitle('Unit - '+unit, fontsize=14)
        plt.tight_layout()
        
        # Save png
        path='../../09-Results/Stage-01/plot_LocalLinearFit_SlopeScatter/'
        fname=path+unit+'.png'
        plt.savefig(fname,dpi=200)
        


def plot_fit_allslopes(df,df_lreg,df_GL,scale=False):
    # List of locations
    loclist = df_lreg['ID'].unique()
    unitlist = df_lreg['unit_geo'].dropna().unique()  
    
    # loop over units 
    for unit in sorted(unitlist):
        f,axs=plt.subplots(1,3,figsize=(14,10))
        for ii, feature in zip([0, 1, 2],['qc', 'fs', 'u2']):
            df_data=df.loc[(df['unit_geo']==unit),['z_bsf',feature]].dropna()
            axs[ii].plot(df_data[feature],df_data['z_bsf'],'.')
            
            # plot global regression
            sl_GL=df_GL.loc[(df_GL['feature']==feature)  & (df_GL['unit_geo']==unit),'slope'].values[0]
            inter_GL=df_GL.loc[(df_GL['feature']==feature)  & (df_GL['unit_geo']==unit),'intercept'].values[0]    
            axs[ii].plot(linear_fit(df_data['z_bsf'],sl_GL,inter_GL),df_data['z_bsf'],linewidth=5)
            
            # plot local regression
            df_loc=df_lreg.loc[(df_lreg['unit_geo']==unit) & (df_lreg['feature']==feature)]
            for loc in loclist:
                df_data=df_loc.loc[(df_loc['ID']==loc)]
                if not df_data.empty:
                    zz=df.loc[(df['unit_geo']==unit) & (df['ID']==loc)]['z_bsf'].values
                    axs[ii].plot(linear_fit(zz,df_data['slope'].values[0],
                                            df_data['intercept'].values[0]),zz,linewidth=5)
  
            yminmax=[0,80]
            if feature=='qc':
                xminmax=[0, 100]
            elif feature=='fs':
                xminmax=[-0.2, 3.5]
            elif feature=='u2':
                xminmax=[-1, 5]
            
            axs[ii].grid(True)
            if scale:
                axs[ii].set_xlim(xminmax)
                axs[ii].set_ylim(yminmax)
            axs[ii].set_title(feature)
            axs[ii].invert_yaxis()
            
        axs[0].set_ylabel('Depth below seafloor (m)')
        
        f.suptitle('Unit - '+unit, fontsize=14)
        plt.tight_layout()
        
        # Save png
        path='../../09-Results/Stage-01/plot_LocalLinearFit_AllSlopes/'
        if scale:
            fname=path+'scale_'+unit+'.png'
        else:
            fname=path+unit+'.png'
        plt.savefig(fname,dpi=200)
    

def plot_thickness_ID(df):
    # List of locations
    loclist = df['ID'].unique()
    unitlist = df['unit_geo'].dropna().unique()  
    # loop over units 
    for unit in sorted(unitlist):
        df_zbsf_global=df.loc[df.unit_geo==unit]['z_bsf'].values
        i=0
        
        f=plt.figure(figsize=(12,10),constrained_layout=True)
        gs=f.add_gridspec(2,2)
        ax0 = f.add_subplot(gs[:,0])
        ax1 = f.add_subplot(gs[0,1])
        ax0.plot(np.ones(len(df_zbsf_global))*i,df_zbsf_global,'.')
        z_spread=[]
        for loc in loclist:
            df_zbsf_loc=df.loc[(df.unit_geo==unit) & (df.ID==loc)]['z_bsf'].values
            if len(df_zbsf_loc)>1:
                i+=1
                ax0.plot(np.ones(len(df_zbsf_loc))*i,df_zbsf_loc,'.')
                z_spread.append(df_zbsf_loc.max()-df_zbsf_loc.min())
        
        # print(z_spread)
        # ax0.set_ylim(-1,81)
        ax0.invert_yaxis()
        ax0.set_ylabel('Depth [m]')
        ax1.hist(z_spread)
        ax1.set_xlim(-1,81)
        extent=np.round(df_zbsf_global.max()-df_zbsf_global.min(),2)
        f.suptitle(unit,fontsize=15)
        plt.figtext(0.515,0.45,'Max extension of the unit: '+str(extent)+' m',fontsize=14)
        # Save png
        path='../../09-Results/Stage-01/plot_CPTs_extension/'
        fname=path+unit+'.png'
        plt.savefig(fname,dpi=200)
        
    



#%% Main function to plot Linear fit results
path_database = '../../09-Results/Stage-01/Database.csv'
# path_dict = '../../09-Results/Stage-01/LocalLinearFit-Kfold-results.pkl'
path_dict = '../../09-Results/Stage-01/LocalLinearFit-Kfold-results_bounds.pkl'
# Load Global linear regrssion results
path_global = '../../09-Results/Stage-01/GlobalLinearFit-Kfold-Results.pkl'


df_database = pd.read_csv(path_database)
df_lreg = load_obj(path_dict)
df_GL = load_obj(path_global)

# Define xmin/xmax for each plot
qcmin, qcmax = 0, 100
fsmin, fsmax = -1, 5
u2min, u2max = -1, 5
Icmin, Icmax = 0, 4
df_xminmax = pd.DataFrame({'qc':[qcmin, qcmax], 'fs':[fsmin, fsmax],
                            'u2':[u2min, u2max], 'ICN':[Icmin, Icmax]})

# path_pdf = '../../09-Results/Stage-01/LocalLinearFit-AllCPTs-LinearFit_bounds.pdf'
# plot_LocalLinFit(df_database, df_lreg, path_pdf, df_xminmax,pdfpng='pdf')

path_png = '../../09-Results/Stage-01/plot_LocalLinearFits_AllCPTs_bounds/'
# plot_LocalLinFit(df_database, df_lreg, path_png, df_xminmax,pdfpng='png')

path_pdf = '../../09-Results/Stage-01/LocalLinearFit-AllCPTs-LinearFit_KfoldMetrics_tmp.pdf'
# plot_LocalLinFit_Metrics(df_database, df_lreg, path_pdf, df_xminmax)

# plot_fit_spread(df_lreg, df_GL)
# plot_fit_allslopes(df_database,df_lreg,df_GL,scale=False)
# plot_fit_allslopes(df_database,df_lreg,df_GL,scale=True)
# plot_thickness_ID(df_database)



