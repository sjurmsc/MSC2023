# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 10:58:39 2021

Global linear fit model
Perform linear regression per unit for the entire dataset

@author: GuS
"""

#%% Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages
import pickle

from scipy.stats import norm
from sklearn import linear_model
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_validate, cross_val_predict
from scipy import signal
from sklearn.base import RegressorMixin
from sklearn.base import BaseEstimator 

import GM_Toolbox as GMT 

#%% Define functions
def save_obj(path_obj, obj):
    with open(path_obj, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
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
    ax.grid(True)
    return ax


def plot_data_fit(ax, y, X, y_pred, feature, xminmax, yminmax):
    y=y.flatten()
    y_pred=y_pred.flatten()
    X=X.flatten()

    # new way, based on interquartile range on the error
    y_true0,y_pred0=data_preparation(X,y,y_pred,detrend='quartile',bounds=[-1, 1])
    
    # old way, based on normalized erros and fix range
    # errors = y_pred[y!=0] - y[y!=0]
    # model_dist=errors/ y[y!=0]    
    # y_true0=y[y!=0][(model_dist>-1.5) & (model_dist<1.5)]
    # y_pred0=y_pred[y!=0][(model_dist>-1.5) & (model_dist<1.5)]  

    _, mae, accuracy, mu, std,mape = GMT.evaluate_modeldist(y_true0, y_pred0)
    # std=np.min((0.5,std))
    
    ax.plot(y, X, '.', markersize=1)
    ax.plot(y_pred, X, 'k')
    ax.set_ylabel('Depth below seafloor (m)')
    
    yminmax=[0,80]
    if feature=='qc':
        xminmax=[0, 100]
    elif feature=='fs':
        xminmax=[-0.2, 3.5]
    elif feature=='u2':
        xminmax=[-1, 5]
        
    ax = plot_axis(ax1,'$%s_%s$ (MPa)'%(feature[0], feature[1]), xminmax, yminmax)
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    
    lin_reg = linear_model.LinearRegression()
    lin_reg.fit(X.reshape(-1,1),y_pred.reshape(-1,1))
    a, b = lin_reg.coef_, lin_reg.intercept_
    
    X_std = np.array([X.min(), X.max()])
    y_std = np.array([y_pred[(X==X_std[0])][0], y_pred[(X==X_std[1])][0]])
    

    # if a>0:
    #     y_std = np.array([y_pred.min(), y_pred.max()])
    # else:
    #     y_std = np.array([y_pred.max(), y_pred.min()])
    
    # ax.fill_betweenx(X_std, y_std*(1-std), y_std*(1+std), alpha=0.2, color='k', edgecolor='k')
    ax.fill_betweenx(X_std, y_std-std, y_std+std, alpha=0.2, color='k', edgecolor='k')
    
    
    fit_txt = ('Linear fit\nSlope: %.4f\nIntercept: %.4f' %(a, b))
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                          edgecolor='none', linewidth=0)
    ax.legend([extra], [fit_txt], loc='lower right')
    # if feature=='qnet':
    #     ax.set_xscale('log')
    return ax



def plot_regression_results(fig, ax, y_true, y_pred, feature, unit, n_splits, scores=[], dense=False):    
    y_pred=y_pred.flatten()
    y_true=y_true.flatten()
    
    # new way, based on interquartile range on the error
    y_true0,y_pred0=data_preparation(X,y_true,y_pred,detrend='quartile',bounds=[-1, 1])
    
    # old way, based on normalized erros and fix range
    # errors = y_pred[y_true!=0] - y_true[y_true!=0]
    # model_dist=errors / y_true[y_true!=0]
    # y_true0=y_true[y_true!=0][(model_dist>-1.5) & (model_dist<1.5)]
    # y_pred0=y_pred[y_true!=0][(model_dist>-1.5) & (model_dist<1.5)]


    _, mae, accuracy, mu, std,mape = GMT.evaluate_modeldist(y_true0, y_pred0)
    # std=np.min((0.5,std))
    if not scores:
        # scores_txt = ('\n' + r'$MAE={:.2f}$' + '\n' + r'$Accuracy={:.2f}$ %').format(mae, accuracy)
        scores_txt = ('\n' + r'$MAE={:.2f}$').format(mae)
    else:
        mae, mae_std = scores[0], scores[1]
        # scores_txt = ('\n' + r'$MAE={:.2f} \pm {:.2f}$' + '\n' + r'$Accuracy={:.2f}$ %').format(mae, mae_std, accuracy)
        scores_txt = ('\n' + r'$MAE={:.2f} \pm {:.2f}$' ).format(mae, mae_std)
    """Scatter plot of the predicted vs true targets."""
    if dense:
        hb = ax.hexbin(y_true, y_pred, gridsize=40, bins='log', alpha=0.5, edgecolor=None,
                        extent=(y_true.min(), y_true.max(), y_pred.min(), y_pred.max()), cmap='bone_r')
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('counts')  
    else:
        ax.plot(y_true, y_pred, '.', alpha=0.2,  color='tab:blue', markersize=5)
    
    # Square plot
    # ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],'--k', linewidth=2)
    # ax.plot([y_true.min(), y_true.max()], [y_true.min()+std, y_true.max()+std],':k', linewidth=2)
    # ax.plot([y_true.min(), y_true.max()], [y_true.min()-std, y_true.max()-std],':k', linewidth=2)
    
    # better focus on the data, but not true perspective
    ax.plot([y_pred.min(), y_pred.max()], [y_pred.min(), y_pred.max()],'--k', linewidth=2)
    ax.plot([y_pred.min()+std, y_pred.max()+std], [y_pred.min(), y_pred.max()],':k', linewidth=2)
    ax.plot([y_pred.min()-std, y_pred.max()-std], [y_pred.min(), y_pred.max()],':k', linewidth=2)
    
    # if ax.get_ylim()[0]>0:
    #     ax.set_ylim([0, ax.get_ylim()[1]])
    
    ax.set_xlabel('Measured $%s_{%s}$' %(feature[0], feature[1]))
    ax.set_ylabel('Predicted $%s_{%s}$' %(feature[0], feature[1]))
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                          edgecolor='none', linewidth=0)
    ax.legend([extra], [scores_txt], loc='upper left', prop={'size': 10})    
    title = 'Group %s-fold Cross-validation results' %(n_splits)
    ax.set_title(title)
    ax.grid(True)
    # ax.axis('equal')
    return ax


def plot_histogram_results(ax, y_true, y_pred, feature, unit):
    y_true=y_true.flatten()
    y_pred=y_pred.flatten()
    
    # new way, based on interquartile range on the error
    y_true0,y_pred0=data_preparation(X,y_true,y_pred,detrend='quartile',bounds=[-1, 1])
    
    # old way, based on normalized erros and fix range
    # errors = y_pred[y_true!=0] - y_true[y_true!=0]
    # model_dist=errors / y_true[y_true!=0]
    # y_true0=y_true[y_true!=0][(model_dist>-1.5) & (model_dist<1.5)]
    # y_pred0=y_pred[y_true!=0][(model_dist>-1.5) & (model_dist<1.5)]

    _,mae,accuracy,mu,std,mape = GMT.evaluate_modeldist(y_true0,y_pred0)
    # std=np.min((0.5,std))

    # n, bins, patches = ax.hist(y_pred - y_true, 50, edgecolor='black', density=True, range=[-2.5, 2.5], facecolor='green', alpha=0.5)
    # x = np.linspace(-2.5, 2.5, 100)
    n, bins, patches = ax.hist(y_pred - y_true, 50, edgecolor='black', density=True, range=[-5*std, 5*std], facecolor='green', alpha=0.5)
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
    ax.legend([extra],['$\mu=%.3f$\n$\sigma=%.3f$' %(mu,std)],loc='upper left',prop={'size':10})
    ax.grid(True)
    
    labels=['-4*$\sigma$','-3*$\sigma$','-2*$\sigma$','-$\sigma$',0,'$\sigma$','2*$\sigma$','3*$\sigma$','4*$\sigma$']
    ax.set_xticks([-4*std, -3*std, -2*std, -std, 0, std, 2*std, 3*std, 4*std])
    ax.set_xticklabels(labels)
    return ax



def data_preparation(X,y,y_pred=0,detrend=0,bounds=[-1, 1]):
    if type(detrend) is int:
        y_d=signal.detrend(y.flatten())
        y_d=signal.detrend(y_d)
        std_d=np.nanstd(y_d)
        median_d=np.nanmedian(y_d)
        y_d=y_d-median_d
        X0=X[(y_d>bounds[0]*std_d) & (y_d<bounds[1]*std_d)] 
        y0=y[(y_d>bounds[0]*std_d) & (y_d<bounds[1]*std_d)].flatten()
        return X0, y0
    
    elif detrend=='quartile':
        errors = y_pred - y
        q25,q75=np.percentile(errors,[25,75])
        intqrt=q75-q25
        y0=y[(errors>(q25-1.5*intqrt)) & (errors<(q75+1.5*intqrt))]
        y_pred0=y_pred[(errors>(q25-1.5*intqrt)) & (errors<(q75+1.5*intqrt))]
        return y0, y_pred0
    
    elif detrend=='detrend':
        y_d=y.flatten()-y_pred.flatten()
        std_d=np.nanstd(y_d)
        X0=X[(y_d>bounds[0]*std_d) & (y_d<bounds[1]*std_d)] 
        y0=y[(y_d>bounds[0]*std_d) & (y_d<bounds[1]*std_d)].flatten()
        y_pred0=y_pred[(y_d>bounds[0]*std_d) & (y_d<bounds[1]*std_d)]
        return X0, y0, y_pred0



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


class RTK_Regressor_old(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        # The prediction of simple model 
        lin_Hub = linear_model.HuberRegressor(epsilon=1.35)
        z  = X       
        # Filtering to avoid to big influence of sand lines
        X0,y0=data_preparation(X,y,0,detrend=0,bounds=[-1, 1])

#        self.params_ = np.polyfit(z0, y0,1)[:,0]
        self.model  = lin_Hub.fit(X0,y0)
        self.coef_   = lin_Hub.coef_
        self.intercept_   = lin_Hub.intercept_
        # self.coef_   = model.coef_
        # self.intercept_   = model.intercept_
        return self
    
    def predict(self, X):
        return self.model.predict(X)
#        return self._model(X[:,0], self.params_)


#    def _model(self, x, params):  
#      
#        line = np.poly1d(self.params_)(x)
#        line[line<0]=0
#        line[line>85]=85
#        return line
#    

#def rtk_fit(df_data, feature): 
#    gam=19.5
#    a=0.55
#    z=df_data['z_bsf'].values
#    qc=df_data['qc'].values
#    fs=df_data['fs'].values
#    u2=df_data['u2'].values
#    qt=(qc*1000+u2*(1-a))/1000     
#    (Qt,Fr,Bq)=GMT.CPT_norm(z,qt*1000,fs*1000,u2*1000)
#    (sig_vo,sig_vo_eff)=GMT.calc_vertical_stress(z,gam)
#    qnet=(qt*1000-sig_vo)/1000
#
#    
#    Icn_median=np.nanmedian(Icn)    
#    
#    z0=z[(Icn_median-0.5<Icn) & (Icn<Icn_median+0.5)].reshape(-1,1)
#    qc0=qc[(Icn_median-0.5<Icn) & (Icn<Icn_median+0.5)].reshape(-1,1)
#   
#    
#    lin_reg = linear_model.LinearRegression(fit_intercept=False)
#    lin_reg.fit(z0,qc0)
#    y_pred=lin_reg.predict(z.reshape(-1,1))  
#                
#                
##    if feature == 'qc':
##        y_pred=(np.nanmedian(qnet0*1000)+sig_vo)/1000
##    elif feature == 'fs':
##        y_pred=(np.nanmedian(fs*1000)+y*0)/1000
##    elif feature == 'u2':
##        y_pred=(np.nanmedian(u2*1000-z*10)+z*10)/1000
##    y_pred.reshape(-1,1)
#    return (y_pred)



#%% Load database
path_database = '../../09-Results/Stage-01/Database.csv'
df = pd.read_csv(path_database)

z_min, z_max = -1, 80
# z_min, z_max = df['z_bsf'].min(), df['z_bsf'].max()
df = df[(df['z_bsf']>=z_min) & (df['z_bsf']<=z_max)]

#%% Global linear fit per unit
# Define xmin/xmax for each plot
qcmin, qcmax = 0, 100
fsmin, fsmax = -1, 5
u2min, u2max = -1, 5
Qtmin, Qtmax = 1, 100
Frmin, Frmax = -1, 5
Bqmin, Bqmax = -1, 5
df_xminmax = pd.DataFrame({'qc':[qcmin, qcmax], 'fs':[fsmin, fsmax], 'u2':[u2min, u2max],
                           'qnet':[Qtmin, Qtmax], 'Fr':[Frmin, Frmax], 'Bq':[Bqmin, Bqmax]})

# create a PdfPages object
pdfpng='png'
if pdfpng=='pdf':
    pdf = PdfPages('../../09-Results/Stage-01/GlobalLinearFit-Results_20210906_TMP.pdf')


unitlist = df['unit_geo'].dropna().unique()
df_store = pd.DataFrame([])
for unit in sorted(unitlist):
    df_tmp = pd.DataFrame(columns=['feature', 'x', 'y', 'unit', 'lreg', 'slope', 'intercept', 'std']) #, 'r2', 'mae', 'r2_std', 'mae_std'])
    print('Unit: ', unit)
    
    # To set intercept to 0 for unit0XX
    # if unit[-1].isdigit():
    #     Ulvl=unit[-2:-1]
    # else:
    #     Ulvl=unit[-3:-2]
    Ulvl=1
    # if Ulvl is = 0 , then the intercept is 0
    for feature in ['qc', 'fs', 'u2']:
        # if feature=='u2':
        #     Ulvl=1
        df_data = df.loc[df['unit_geo']==unit, ['ID','z_bsf', 'x', 'y', feature]].dropna(subset=[feature])
        X = df_data[['z_bsf']].values
        y = df_data[feature].values.reshape(-1,1)
        groups=df_data['ID']

        # Instantiate linear regression model 
        lin_reg = RTK_Regressor()
        # lin_reg = RTK_Regressor_old()

        # Evaluation of Cross validation score
        n_splits = np.min([5,df_data['ID'].unique().shape[0]])
        if n_splits > 1:
            cv = GroupKFold(n_splits=n_splits)
            score = cross_validate(lin_reg, X, y, groups=groups, cv=cv, scoring=['r2', 'neg_mean_absolute_error'], n_jobs=1, verbose=0)
            y_predcross = cross_val_predict(lin_reg, X, y, groups=groups, cv=cv, n_jobs=1, verbose=0)
            # scores_txt = (r'$R^2={:.2f} \pm {:.2f}$' + '\n' + r'$MAE={:.2f} \pm {:.2f}$').format(np.mean(score['test_r2']),
            #                                                                        np.std(score['test_r2']),                                                                             np.std(score['test_neg_mean_absolute_error']))
            scores = [-np.mean(score['test_neg_mean_absolute_error']), np.std(score['test_neg_mean_absolute_error'])]
            
            lin_reg.fit(X,y)
            y_pred = lin_reg.predict(X).reshape(-1,1)
            if feature=='qc':
                if lin_reg.coef_[0]<0:
                    lin_reg.coef_=np.array([0.0001])
                    lin_reg.model.coef_=np.array([0.0001])
                    lin_reg.intercept_=np.mean(np.unique(y_pred))
                    lin_reg.model.intercept_=np.mean(np.unique(y_pred))
                    y_pred = lin_reg.predict(X).reshape(-1,1)
                    
            
            # X0,y_true0,y_pred0=data_preparation(X,y,y_pred=0,detrend='detrend',bounds=[-1, 1])
            y_true0,y_pred0=data_preparation(X,y,y_pred,detrend='quartile',bounds=[-1, 1])
   
            model_dist,mae,accuracy,mu,std,mape = GMT.evaluate_modeldist_norm(y_true0, y_pred0)
            # model_dist,mae,accuracy,mu,std,mape = GMT.evaluate_modeldist(y, y_pred)
            

            # Store results to dataframes
            df_tmp.loc[0, 'feature'] = feature
            df_tmp.loc[0, 'unit_geo'] = unit
            # df_tmp.loc[0, 'lreg'] = lin_reg
            df_tmp.loc[0, 'slope'] = lin_reg.coef_[0]
            df_tmp.loc[0, 'intercept'] = lin_reg.intercept_
            df_tmp.loc[0, 'x'] = df_data['x'].mean()
            df_tmp.loc[0, 'y'] = df_data['y'].mean()
            df_tmp.loc[0, 'std'] = std
                                     
            # Plot outputs
            fig = plt.figure(figsize=(10,9))
            fig.suptitle('Global Linear regression for $%s_{%s}$ - Unit %s' %(feature[0], feature[1], unit), fontsize=14)
            gs = GridSpec(2, 2, width_ratios=[2,3])
            ax1 = fig.add_subplot(gs[:,0])
            ax2 = fig.add_subplot(gs[0,-1])
            ax3 = fig.add_subplot(gs[1,-1])
            
            ax1 = plot_data_fit(ax1, y, X[:,0].reshape(-1,1), y_pred, feature, np.array([0,y.max()]), [X.min(),X.max()])
            ax2 = plot_regression_results(fig, ax2, y, y_pred, feature, unit, n_splits, scores=scores, dense=True)
            ax3 = plot_histogram_results(ax3, y, y_pred, feature, unit)
            
            plt.tight_layout()
            
        df_store = df_store.append(df_tmp,ignore_index=True)
        # Save to PDF    
        if pdfpng=='png':
            # path_png='../../09-Results/Stage-01/plot_GlobalLinearFit-Results/'
            path_png = '../../09-Results/Stage-01/Final/Plots_GlobalLinear/'
            fname=path_png + 'GL-Unit_' + unit + '_' + feature + '.png'
            plt.savefig(fname,dpi=200)
        if pdfpng=='pdf':
            pdf.savefig(fig)
if pdfpng=='pdf':
    pdf.close()  
        
#%% Save results to pickle
path_dict = '../../09-Results/Stage-01/GlobalLinearFit-Kfold-Results.pkl'
save_obj(path_dict, df_store)             
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        