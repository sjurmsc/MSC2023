# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 16:48:08 2019

Structural modelling: Build structural model from seismic horizons
Toolbox

@author: GuS
"""

#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
from scipy.optimize import least_squares

#%% Evaluate model distribution
def evaluate_modeldist_norm(y_true, y_pred):
    errors = y_pred - y_true
    model_dist=errors[y_true!=0] / y_true[y_true!=0]
    mape = 100 * np.mean(np.abs(model_dist))
    mae = np.mean(np.abs(errors))
    accuracy = np.abs(100 - mape)
    mu=np.mean(model_dist[(-2.5<model_dist) & (model_dist<2.5)])
    std = np.std(model_dist[(-2.5<model_dist) & (model_dist<2.5)])
    model_dist = model_dist[(-2.5<model_dist) & (model_dist<2.5)]
    return model_dist, mae, accuracy, mu, std, mape


def evaluate_modeldist(y_true, y_pred):
    errors = y_pred - y_true
    # perc = 100*y_pred[y_true!=0]/y_true[y_true!=0]
    model_dist=errors
    mape = 100 * np.mean(np.abs(model_dist))
    mae = np.mean(np.abs(errors))
    accuracy = np.abs(100 - mape)
    mu=np.mean(model_dist) 
    std = np.std(model_dist) 
    # mu=np.mean(model_dist[(-2.5<model_dist) & (model_dist<2.5)])
    # std = np.std(model_dist[(-2.5<model_dist) & (model_dist<2.5)])
    # model_dist = model_dist[(-2.5<model_dist) & (model_dist<2.5)]
    return model_dist, mae, accuracy, mu, std, mape


#%% Upscale (or down sample) CPT data
def upscale_cpt(dfall, upscale=1, n=50):
    dfUall = pd.DataFrame([])
    for borehole in np.unique(dfall['borehole']):
        df = dfall[dfall['borehole']==borehole]
        df = df.sort_values(by=['MD'])
        if upscale:
            # Apply filter
            z0     = np.copy(df['MD'].values)
            qc0    = np.copy(df['qc'].values)
            unt    = np.copy(df['unit'].values)
            inx    = np.linspace(0,len(z0)-1,len(z0)).astype(int)
            qcf    = np.interp(z0,moving_average(z0, n),moving_average(qc0, n))            
            MDU    = np.arange(df['MD'].min(), df['MD'].max(), upscale)
            qc_u   = np.interp(MDU,z0,qcf)
            inx_u  = np.interp(MDU,z0,inx).astype(int)
            unt_u  = unt[inx_u]
            dfU    = pd.DataFrame(data={'borehole': borehole, 'MDU': MDU, 'qcU': qc_u, 'UNITU':unt_u})
            
            dfm    = pd.merge_asof(df, dfU, left_on=['MD'], right_on=['MDU'])
            dfm    = dfm.drop_duplicates(subset=['borehole_x','MDU'], keep='first')
            dfUall = dfUall.append(dfm)           
    dfUall = dfUall.drop(columns=['borehole_y', 'MD', 'qc' ])
    dfUall = dfUall.rename(columns={"borehole_x": "borehole", "MDU": "MD", "qcU": "qc"})
    return dfUall
	
#%% Caluculate mean normalised CPT tip resistance
def norm_cpt(df):
    dfn=df.copy()
    dfn['Qt'] = np.nan
    for unit in np.unique(df['unit'])[:-2]:
        for cpt in pd.unique(df['borehole']):
            z_part=dfn['MD'][(dfn['unit']==unit) & (dfn['borehole']==cpt)]
            qc_part=dfn['qc'][(dfn['unit']==unit) & (dfn['borehole']==cpt)]
            QTpart=(qc_part*1000-20*z_part)/(10*z_part)
            dfn['Qt'][(dfn['unit']==unit) & (dfn['borehole']==cpt)]=np.mean(QTpart[z_part>2])+np.ones(len(z_part))
    dfn['Qt'][~np.isfinite(dfn['Qt'])]=0
    return(dfn)

def CPT_norm(z,qt,fs,u2):
    ''' Normalised the CPT response based on a a constant unit weigth of 9.5 kN/m3 '''
    gam  = 19.5 + z*0
    gamw = 10
    (sig_vo,sig_vo_eff) = calc_vertical_stress(z,gam)
    qnet = qt-sig_vo              # The tip resitance minus total stress
    sig_vo_eff[sig_vo_eff<1]=1
    Qt   = (qnet)/(sig_vo_eff)    # Normalised tip reesistance
    Qt[Qt<1]=1
    Fr   = fs/(qnet)*100          # Normalised friction in %
    Fr[Fr<1]=1
    Bq   = (u2-z*gamw)/(qnet)     # Normalised pore pressure
    return(np.array(Qt),np.array(Fr),np.array(Bq))

#%% CPT fit with contrains
def q_min(z):
    ''' Calculates the minimium accetable tip resistance value based on normalconsolidated clay '''
    gam      = 15
    sigv     = z*gam
    sigv_eff = z*(gam-10)
    su_min   = 0.3*sigv_eff
    Nkt      = 15
    q_min    = su_min * Nkt + sigv
    return(q_min/1000)

def q_max(z):
    ''' Calculates the maximum accetable tip resistance value based on dense sand '''
    DR       = 1.75
    gam      = 20
    sigv_eff = z*(gam-10)
    C0       = 157
    C1       = 0.55
    C2       = 2.41
    q_max    = np.exp(DR*C2)*C0*sigv_eff**C1
    return(q_max/1000)

def cpt_model_cal(z,z0,z1,dq0,dq1):
    ''' Calculates the tip resistance in the given layer '''
    q_min0   = q_min(z0)
    q_min1   = q_min(z1)
    q0       = q_min0+dq0
    q1       = q_min1+dq1
    q        = q0 + (z-z0) * (q1-q0) / (z1-z0)
    return(q)

def cpt_model(z,z0,z1,a,b):
    ''' Calculates the tip resistance in the given layer '''
    dq0,dq1  = int_slp_r(z0,z1,a,b)
    q        = cpt_model_cal(z,z0,z1,dq0,dq1)
    return(q)

def int_slp(z0,z1,dq0,dq1):
    ''' Calculate the global slope (a) and intersept (b) for the cpt_model'''
    y0       = q_min(z0) + dq0
    y1       = q_min(z1) + dq1
    a        = (y1-y0) / (z1-z0)
    b        = y1 - a * z1
    return(a,b)

def int_slp_r(z0,z1,a,b):
    ''' Calculate the local q factors based on the global slope (a) and intersept (b) for the cpt_model'''
    q1       = b + a * z1
    q0       = q1 - a * (z1-z0)
    dq0      = q0 - q_min(z0)
    dq1      = q1 - q_min(z1)
    return(dq0,dq1)

def residual(C,q_data,z_data,z0,z1):
    ''' Calculates the diffrence between the data and the prediction model '''
    dq0      = C[0]
    dq1      = C[1]
    q_model  = cpt_model_cal(z_data,z0,z1,dq0,dq1)
    res      = q_data-q_model
    return(res)

def calibrate_cpt_model(q_data,z_data,z0,z1):
    ''' Calibrates the linear model to the data using least sqare resression '''
    constant = np.ones(2)
    bound_min= np.array([0,0])
    bound_max= np.array([q_max(z0) - q_min(z0),q_max(z1) - q_min(z1)])
    if z0==0:
        bound_max= np.array([1,q_max(z1)  - q_min(z1)])
    cpt_lsq  = least_squares(residual, constant, bounds=(bound_min, bound_max), diff_step=1, verbose=0, args=(q_data,z_data,z0,z1))
    C        = cpt_lsq.x
    dq0      = C[0]
    dq1      = C[1]
    if z0==0:
        dq0  = 0
    (a,b)    = int_slp(z0,z1,dq0,dq1)
    return(a,b)

def calibrate_cpt_model_clean(qt_all,z_all):
    ''' Function that removes 2 standard deviation for calibration '''
    z0=z_all[0]
    z1=z_all[-1]
    clean=np.isfinite(qt_all)
    qt_all_clean=qt_all[clean]
    z_all_clean=z_all[clean]
    cc=calibrate_cpt_model(qt_all_clean,z_all_clean,z0,z1)
    return(cc,qt_all_clean,z_all_clean)

#%% Calculate soil behaviur type
def calc_Icn(qt,fs,sig_vo,sig_vo_eff,n):
    fs[fs<0.01]=0.01
    qt[qt<0.01]=0.01
    patm=100
    Qtn=((qt-sig_vo)/patm ) * (patm/sig_vo_eff)**(n)
    Fr=fs/(qt-sig_vo)*100
    Qtn[Qtn<0.01]=0.01
    Fr[Fr<0.01]=0.01
    A=3.47-np.log10(Qtn)
    B=1.22+np.log10(Fr)
    Icn=np.sqrt(A**2+B**2)
    nn=0.381*Icn+0.05*(sig_vo_eff/patm)-0.15
    return(Icn,nn)

def calc_vertical_stress(z,gam):
    ''' Function that calcualtes the vertical effective and total stress
    input: z (array) [m], gam (array or constant) [kN/m3]
    output: sig_vo [kN/m2],sig_vo_eff [kN/m2]
    '''
    gamw=10 # unit weight water
    
    if np.size(gam)>1:
        if z[0]<0.1:
            z[0]=0.01
        sig_vo=np.append(z[0]*gam[0],np.cumsum(np.diff(z)*gam[1:]))
        sig_vo_eff=np.append(z[0]*(gam[0]-gamw),np.cumsum(np.diff(z)*(gam[1:]-gamw)))
    else:
        if np.size(z)>1:
            z[z<0.1]=0.1
        else:
            if z<0.1:
                z=0.01
        sig_vo=z*gam
        sig_vo_eff=z*(gam-gamw)
    return(sig_vo,sig_vo_eff)
#%% Evaluation plot

def evaluate(y_pred, y, model_set,folder):
    from scipy.stats import norm
    errors = y_pred - y
    model_dist=errors[y!=0] / y[y!=0]

    mape = 100 * np.mean(np.abs(model_dist))
    accuracy = 100 - mape

    mu=np.mean(model_dist[(-1<model_dist) & (model_dist<1) ])
    stdv = np.std(model_dist[(-1<model_dist) & (model_dist<1) ])
    t_max=y.max()
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(9,4))
    n, bins, patches=ax1.hist(model_dist, 50, density=True, range=[-1.5, 2.5] , facecolor ='orange', label='model val.')
    ax1.plot(np.linspace(-1.5, 2.5,100),norm.pdf(np.linspace(-1.5, 2.5,100),mu,stdv),color='k')
    ax1.grid(color="0.5", linestyle=':', linewidth=0.5)
    ax1.set_xlabel('$(Y_{pred}-Y)/Y$')
    ax1.set_ylabel('Frequency')
    ax1.set_title(r'$\mathrm{Data:}\ \mu=%.1f,\ \sigma=%.2f$' %(mu,stdv))
    sf=0.9
    ax1.text(np.min(bins)*0.9,np.max(n)*sf, '{:5s} {:s} '.format(model_set, 'set'), fontsize=14)
    ax2.scatter(y, y_pred, color='orange',s=10,alpha=0.01)
    ax2.plot([0,t_max],[0,t_max], color="0.2", linestyle=':', linewidth=1.5)  
    ax2.plot([0,t_max],[0,t_max*(1+stdv)], color="0.2", linestyle=':', linewidth=1.5) 
    ax2.plot([0,t_max],[0,t_max*(1-stdv)], color="0.2", linestyle=':', linewidth=1.5) 
    ax2.set_xlabel("Measured$")  
    ax2.set_ylabel("Prediction")    
    ax2.grid(color="0.5", linestyle=':', linewidth=0.5)
    ax2.text(t_max*0.1, t_max*(1+stdv)*0.9, 'Model Performance', fontsize=10)
    ax2.text(t_max*0.1, t_max*(1+stdv)*0.8, 'Average Error: {:0.0f} kN.'.format(np.mean(errors)), fontsize=10)
    ax2.text(t_max*0.1, t_max*(1+stdv)*0.7, 'Accuracy = {:0.2f}%.'.format(accuracy), fontsize=10)   
    plt.tight_layout()
    plt.savefig(str(folder)+'HistModel_'+model_set+'.png', dpi=300, bbox_inches='tight')
    return(accuracy,stdv,mu)

from scipy.stats import norm
def eval_2fit(Y,Y_pred1,Y_pred2,name1,name2):
    model_dist1=(Y_pred1[Y>0]-Y[Y>0])/Y[Y>0]
    ss_res = np.sum((Y - Y_pred1) ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    r21 = 1 - (ss_res / ss_tot)
    mu1=np.mean(model_dist1[(model_dist1<2)]) 
    stdv1 = np.std(model_dist1[(model_dist1<2)])
    
    model_dist2=(Y_pred2[Y>0]-Y[Y>0])/Y[Y>0]
    ss_res2 = np.sum((Y - Y_pred2) ** 2)
    ss_tot2 = np.sum((Y - np.mean(Y)) ** 2)
    r22 = 1 - (ss_res2 / ss_tot2)
    mu2=np.mean(model_dist2[(model_dist2<2)]) 
    stdv2 = np.std(model_dist2[(model_dist2<2)])
    
    rnge=2
    t_max=np.max(Y)*1.1
    
    n=plt.hist(model_dist2, 50, density=True, range=[-rnge, rnge] , facecolor ='red',alpha=0.5, label=name1)
    hist_max=np.max(n[0])
    
    fig, axs = plt.subplots(1,2, figsize=(8,4))
    axs[0].hist(model_dist1, 50, density=True, range=[-rnge, rnge] , facecolor ='orange', label=name1)
    axs[0].hist(model_dist2, 50, density=True, range=[-rnge, rnge] , facecolor ='red',alpha=0.5, label=name1)
    axs[0].plot(np.linspace(-rnge, rnge,100),norm.pdf(np.linspace(-rnge, rnge,100),mu1,stdv1),'-k')
    axs[0].plot(np.linspace(-rnge, rnge,100),norm.pdf(np.linspace(-rnge, rnge,100),mu2,stdv2),'--k')
    axs[0].grid(color="0.5", linestyle=':', linewidth=0.5)
    axs[0].set_xlabel('$(Y_{Prediction}-Y_{Measured})/Y_{Measured}$')
    axs[0].set_ylabel('Frequency')
    axs[0].text(-rnge*0.9,hist_max*0.9,r'$\mu_1=$ %.1f, $\sigma_1$=%.2f' %(mu1,stdv1))
    axs[0].text(-rnge*0.9,hist_max*0.8,r'$\mu_2=$ %.1f, $\sigma_2$=%.2f' %(mu2,stdv2))
    axs[0].set_xlim([-rnge,rnge])
    
    axs[1].set_xlabel("$Y_{Measured}$")
    axs[1].set_yticks([])
    axs2 = axs[1].twinx()
    axs2.scatter(Y, Y_pred1, color='orange',s=10,alpha=0.01)
    axs2.scatter(Y, Y_pred2, color='red',s=10,alpha=0.01)
    axs2.scatter([], [], color='orange',s=10,alpha=1,label=name1)
    axs2.scatter([], [], color='red',s=10,alpha=1,label=name2)
    axs2.plot([0,t_max],[0,t_max], color="0.2", linestyle=':', linewidth=1.5)
    axs2.set_xlabel("$Y_{Measured}$")
    axs2.set_ylabel("$Y_{Prediction}$")
    axs2.grid(color="0.5", linestyle=':', linewidth=0.5)
    axs2.legend(loc=4)
    axs2.text(25,t_max*0.9,r'$R_1^2$=%.2f' %(r21))
    axs2.text(25,t_max*0.8,r'$R_2^2$=%.2f' %(r22))
    axs2.set_xlim([0,t_max])
    axs2.set_ylim([0,t_max])
#    fig.savefig(name1+'_'+name2+'.svg', format='svg', bbox_inches='tight')
    fig.savefig(name1+'_'+name2+'.png', dpi=600, bbox_inches='tight')
    return()       

#%% moving average filter
def moving_average(a, n=4):
    if (n % 2) != 0:
        n=n+1
    ma=np.copy(a)
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ma[int(n/2-1):int(-n/2)]=ret[n - 1:] / n
    return ma

#%% Linear fit
def estimate_coef(x, y): # Linear fit
    # number of observations/points
    n = np.size(x)
    # mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)
    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
    return(b_0, b_1)

#%% make soil layering marix
def soil_lay(zi,qci,uniti,m_sand,m_clay,figure_plot):
    clay=np.array([4,6])
    nkt=20
    gam=20
    gamw=10
    pa=100
    gam_eff=np.ones(len(zi))*(gam-gamw)
    sig=zi*gam
    sig_eff=sig-zi*gamw
    qt=qci*1000
    qt1=(qt/pa)/((sig_eff/pa))**0.5
    phic=17.6+11*np.log10(qt1)
    phic[phic>45]=45
    phi=np.rad2deg(np.arctan(np.tan(np.deg2rad(phic))/m_sand))-5
    su=((qt-sig)/nkt)/m_clay
    soil=np.zeros((len(np.unique(uniti)),6))
    j=0
    if figure_plot==1:
        cmap = plt.get_cmap('tab10')
        col = [cmap(i) for i in np.linspace(0, 1, 10)]
        fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4, sharey=True,figsize=(8,8))
        ax1.invert_yaxis()
    z0=0
    for unitij in np.unique(uniti):
        z1=np.max(zi[uniti==unitij])
        zfit=zi[uniti==unitij]
        if unitij in clay:
            phi_lay=0
            if len(zi[uniti==unitij])==1:
                su0_lay = su[uniti==unitij]
                k_lay   = 0
                sufit   = np.mean(su0_lay)+zfit*0
            else:
                b       = estimate_coef(zi[uniti==unitij], su[uniti==unitij])
                su0_lay = np.round(b[0]+b[1]*z0,1)
                k_lay   = np.round(b[1],2)
                sufit   = b[0]+b[1]*zfit
            if figure_plot==1:
                ax3.plot(su[uniti==unitij],zi[uniti==unitij], color=col[int(unitij)])
                ax3.plot(sufit,zfit,'--k')
        else:
            phi_lay=np.round(np.mean(phi[uniti==unitij]),1)
            su0_lay=0
            k_lay=0
            phifit=np.ones(len(zfit))*phi_lay
            if figure_plot==1:
                ax2.plot(phi[uniti==unitij],zi[uniti==unitij], color=col[int(unitij)])
                ax2.plot(phifit,zfit,'--k')
        soil[j,:]=[z0,z1,gam-gamw,phi_lay,su0_lay,k_lay]
        z0=z1
        j+=1
        if figure_plot==1:
            ax1.plot(gam_eff[uniti==unitij],zi[uniti==unitij], color=col[int(unitij)])
            ax4.plot(qt[uniti==unitij]/1000,zi[uniti==unitij], color=col[int(unitij)])
    if figure_plot==1:
        ax1.set_ylabel('Depth [m]')
        ax1.set_xlabel('$\gamma$ [$kN/m^3$]')
        ax2.set_xlabel('$\phi$ [$\degree$]')
#        ax2.set_xlabel('$s_u$ [$MNm$]')
        ax3.set_xlabel('$s_u$ [$MNm$]')
        ax4.set_xlabel('$q_t$ [$MN$]')
        plt.show()
    return(soil)

#%% Make Brinch Hansen calculation
def brinch_hansen_capacity(geo_soil_load, depth):
    '''Function that calculates the vertical bearing capacity according to
    Brinch Hansen (2005)

    Parameters
    ----------
    lx          = Length [m]
    by          = Width [m]
    d           = depth [m]
    gam_eff     = effective unit weight [kN/m2]
    phi         = Angle of friction [degress]
    q           = Overburden load [kN/m2]
    c           = Cohesion [kN/m3]
    Mx          = Moment around length direction [kNm]
    My          = Moment around width direction [kNm]
    Hx          = Horisontal load in length direction [kN]
    Hy          = Horisontal load in width direction  [kN]
    V           = Vertical load  [kN]

    Returns
    -------
    R_A        = bearing pressure [kPa] '''

    lx      = geo_soil_load[0]
    by      = geo_soil_load[1]
    gam_eff = geo_soil_load[2]
    phi     = geo_soil_load[3]
    c       = geo_soil_load[4]
    q       = geo_soil_load[5]
    Mx      = geo_soil_load[6]
    Hx      = geo_soil_load[7]
    My      = geo_soil_load[8]
    Hy      = geo_soil_load[9]
    V       = geo_soil_load[10]

    d       = depth
    phi=np.deg2rad(phi)
    ex=Mx/V
    ey=My/V
    H=np.sqrt(Hx**2+Hy**2)
    l=lx-2*ex
    b=by-2*ey
    A=l*b
    if phi==0:
        Ngam=0
        Nq=1
        Nc=2+np.pi
        ic=0.5+0.5*np.sqrt(1-H/(A*c))
        iq=ic
    else:
        Nq=np.exp(np.pi*np.tan(phi))*(1+np.sin(phi))/(1-np.sin(phi))
        Ngam =1.5*(Nq-1)*np.tan(phi)
        Nc=(Nq-1)/np.tan(phi)
        ic=(1-H/(V+A*c*(1/np.tan(phi))))**2
        iq=ic
    sgam=1-0.4*b/l
    sq=1+0.2*b/l
    sc=sq
    igam=iq**2
    dq=1+0.35*d/b
    dc=dq
    R_A=1/2*gam_eff*b*Ngam*sgam*igam + q*Nq*sq*iq*dq + c*Nc*sc*ic*dc
    return R_A
#%% Spudcan calculation
def SpudcanPen(A,z,soil,V):
    lx      = (4*A/np.pi)**(1/2)*np.ones(len(z))
    by      = (4*A/np.pi)**(1/2)*np.ones(len(z))
    Mx      = np.zeros(len(z))
    Hx      = np.zeros(len(z))
    My      = np.zeros(len(z))
    Hy      = np.zeros(len(z))
    dz      = np.append(0,np.diff(z))
    gam_eff = np.array([soil[:,2][zi<=soil[:,1]][0] for zi in z])
    q       = np.cumsum(gam_eff*dz)
    phi     = np.array([soil[:,3][zi<=soil[:,1]][0] for zi in z])
    c       = np.array([soil[:,4][zi<=soil[:,1]][0]+soil[:,5][zi<=soil[:,1]][0]
    * (zi-soil[:,0][zi<=soil[:,1]][0]) for zi in z])
    geo_soil_load=np.vstack((lx,by,gam_eff,phi,c,q,Mx,Hx,My,Hy,V*np.ones(len(z)))).T
    VV=[]
    for i in  range(0,len(z)):
        VV.append(brinch_hansen_capacity(geo_soil_load[i,:], z[i])*A)
    VV=np.array(VV)
    z_pen=z[VV>V][0]
    return(z_pen)


#%% Seismic-well tie
''' Function to adjust the seismic interpretation to the well tops and merge for a given unit '''
# def SeisWellTie(seisinterp_unit, cptinterp_unit):
#     cpt_unit = cptinterp_unit.copy()
# #    seis_unit = seisinterp_unit.copy()
#     seiswelltie = seisinterp_unit.copy()
#     minhdist = 5     # min horizontal distance to compare CPT interp to seis interp
#     minvdist = 0.25     # min vertical distance to comare CTp interp to seis interp
#     # evaluate distance from cpt to nearest seismic interpretation point
#     for index, row in cpt_unit.iterrows():
#         seiswelltie['hdist'] = distance.cdist(np.array([row.loc[['X','Y']]]), seiswelltie[['X','Y']]).T
#         hdistmin = seiswelltie['hdist'].min()
#         vdistmin = np.abs(seiswelltie.loc[seiswelltie['hdist'].idxmin(),'Z']-row.loc['Z'])
#         if hdistmin >= 0:
#             seiswelltie.append(row.drop(columns=['total_depth', 'MD']))
# #        elif hdistmin < minhdist:
# #            print('Hdistmin:'+ str(hdistmin))
# #            if vdistmin <= minvdist:
# #                print('Vdistmin:'+ str(vdistmin))
# #                seiswelltie.append(row.drop(columns=['total_depth', 'MD']))
# #            elif vdistmin > minvdist:
# #                seiswelltie.drop(seiswelltie[seiswelltie['hdist']<=minhdist].index, inplace=True)
# #                print('Vdistmin:'+ str(vdistmin))
# #                print('DROP')
#     return seiswelltie


