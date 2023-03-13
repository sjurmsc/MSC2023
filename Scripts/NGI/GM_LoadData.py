# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:15:34 2019

Set of functions to load data for subsequent use in the ground model

- load_cpt : function to load CPT data into Pandas dataframe

@author: GuS
"""

#%% Import libraries
import pandas as pd
import numpy as np
import os
import lasio
import matplotlib.pyplot as plt
from scipy.io import netcdf

#%% Load seismic interpretation cloud point
def load_seisinterp_csv(filepath, unit=None, tb=None):
    #UTM84-31N: epsg32631
    df_seis = pd.read_csv(filepath, delim_whitespace=True, header=None, names=['x','y','z'])
    df_seis['unit'] = unit
    df_seis['top/base'] = tb
    df_seis['borehole'] = 'seis'
    return df_seis

#%% Load seismic interpretation from Petrel horizon file
def load_seisinterp_Petrel(filepath, unit=None, tb=None):
    #UTM84-31N: epsg32631
    df_seis = pd.read_csv(filepath, delim_whitespace=True, header=None,
                          index_col=False, names=['x','y','z'], skiprows=9)
    df_seis['unit'] = unit
    df_seis['top/base'] = tb
    df_seis['borehole'] = 'seis'
    return df_seis

#%% Load seismic interpretation from DUG Insight (Mark Vardy)
def load_seisinterp_DUG(filepath, unit=None, tb=None):
    df_seis = pd.read_csv(filepath, delim_whitespace=True, header=None,
                          index_col=False, usecols=[2,3,5], low_memory=False)
    df_seis.columns = ['x','y','z']
    df_seis['unit'] = unit
    # df_seis['top/base'] = tb
    # df_seis['borehole'] = 'seis'
    return df_seis    

#%% Load borehole location
def load_cptloc_csv(filepath):
    # df_cptloc = pd.read_csv(filepath, header=None, delim_whitespace=True,
    #                         names=['borehole', 'x', 'y', 'total_depth'])
    df_cptloc = pd.read_csv(filepath, header=None, delim_whitespace=True,
                            index_col = 0, names=['borehole', 'x', 'y', 'total_depth'])    
    return df_cptloc

#%% Load borehole location from Petrel well head #include water depth!
def load_cptloc_petrel(filepath):
    df_cptloc = pd.read_csv(filepath, sep=' ', header=None, index_col=False, skiprows=35).dropna(axis=1, how='all')
    df_cptloc = df_cptloc[[0,3,4,9,13]]
    df_cptloc.columns = ['borehole','x','y','WD','total_depth']
    return df_cptloc

#%% Load borehole location from Arjen xls file
def load_cptloc_AKT(filepath):
    df_cptloc = pd.read_excel(filepath, sheet_name='Data', header=None, usecols=[7,8,9,10], skiprows=14, names=['borehole','x','y','WD'])
    return df_cptloc

#%% Load well tops from Petrel and merge with location if existing
def load_welltopsPetrel(filepath, cptloc=pd.DataFrame([])):
    df_cptinterp = pd.read_fwf(filepath, header=None, skiprows=1, names=['borehole', 'total_depth', 'unit', 'MD', 'z'])
    
#   # Convert elevation to depth
#    df_cptinterp.Z = df_cptinterp.Z*-1
    
    #Split unit into top/base and unit name
    new = df_cptinterp['unit'].str.split("_", expand=True)
    df_cptinterp['unit'] = new[1]
    df_cptinterp['top/base'] = new[0]
    # Add basement layer at CPT total depth and add water depth
    zmin = np.floor(df_cptinterp['z'].min())
    for bh in df_cptinterp.borehole.unique():
        df_cptinterp = df_cptinterp.append({'borehole' : bh,
                                            'unit' : 'D',
                                            'z' : zmin-5}, ignore_index=True)
    # Merge with cptloc if it existing
    if cptloc.empty:
        print('No CPT loaction defined')
    else:
        df_cptinterp = df_cptinterp.drop('total_depth', axis=1)
        df_cptinterp = pd.merge(cptloc, df_cptinterp, how='inner', on=['borehole'])    
    return df_cptinterp

#%% Load CPT intepretation from xls spreadsheet and merge with cpt location if existing
def load_cptinterp_xls(filepath, cptloc=pd.DataFrame([])):
    df = pd.read_excel(filepath)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
#    unit=df.columns.values[2:11] 
    unit=df.columns.values[1:11] ############### RTK WAS HERE ############################
    df_cptinterp = pd.DataFrame(columns=['borehole','unit','z','MD','a','b'])
    for index, row in df.iterrows():
        borehole = row['Locations']
        WD = row['Water_depth']
        MD = row['Water_depth':'EOCPT'].values
#        z = WD-MD
        z = np.append(WD,WD-MD[1:]) ############### RTK WAS HERE ############################
        ab = row['a_A':]
        a = np.append( np.nan,np.append(ab[::2], np.nan))
        b = np.append(np.nan,np.append(ab[1::2], np.nan))
        df_tmp = pd.DataFrame(data={'unit': unit, 'MD': MD, 'z': z, 'a': a, 'b':b})
        df_tmp['borehole'] = borehole
        df_cptinterp = df_cptinterp.append(df_tmp, ignore_index=True, sort=True)
    # Convert format of to numeric    
    df_cptinterp = df_cptinterp.apply(pd.to_numeric, errors='ignore')
    # Merge with cptloc if it existing
    if cptloc.empty:
        print('No CPT loaction defined')
    else:
        df_cptinterp = pd.merge(cptloc, df_cptinterp, how='inner', on=['borehole'])    
    return df_cptinterp 

#%% Load CPT interp csv and merge with cpt location if exists
def load_cptinterp_csv(filepath, cptloc=pd.DataFrame([])):
    df_cptinterp = pd.read_csv(filepath, sep=';', index_col=0)
    if cptloc.empty:
        print('No CPT loaction defined')
    else:
        df_cptinterp = pd.merge(cptloc, df_cptinterp, how='inner', on=['borehole'])    
    return df_cptinterp 

#%% Load CPT interp csv and merge with cpt location if exists
def load_cptinterp_txt(filepath, cptloc=pd.DataFrame([])):
    df_cptinterp = pd.read_csv(filepath, sep='\t', index_col=0)
    if cptloc.empty:
        print('No CPT loaction defined')
    else:
        df_cptinterp = pd.merge(cptloc, df_cptinterp, how='inner', on=['borehole'])    
    return df_cptinterp 



#%% Load Petrel model
def load_petrel_model(filepath):
    df_model = pd.read_csv(filepath, sep=' ', header=None, names=['i','j','k','x','y','z','units'],
                           index_col=False, skiprows=9)
    return df_model
    

#%% Load CPT data - LAS format
def load_cpts_qc_LAS(directory, cptloc=pd.DataFrame([]), cptinterp=pd.DataFrame([])):
    # Import all LAS in floder into dataframe
    df_qc = pd.DataFrame()
    for filename in os.listdir(directory):        
        if filename.endswith(".las"):
            LAS = pd.read_csv(os.path.join(directory, filename), sep='\s+',
                              header=None, names=['MD', 'qc'], usecols=[0,1], # RTK change from [1,2]
                              index_col=False, skiprows=32)
            LAS['borehole'] = os.path.splitext(filename)[0]
            df_qc = df_qc.append(LAS)
                
    # Merge with cptloc if it exists and convert MD to Z
    if cptloc.empty:
        print('No CPT location defined')
    else:
        df_qc = pd.merge(cptloc, df_qc, how='inner', on=['borehole'])
        df_qc['Z'] = df_qc['WD']-df_qc['MD']

    # Merge with cptinterp if it exists
    if cptinterp.empty:
        print('No CPT interpretation loaded')
    else:
        #remove EOCPT and BaseD0 from interpcpt to be reassigned ############Very specific for TNW ################
        cptinterp = cptinterp[cptinterp['unit']!='EOCPT']
        cptinterp = cptinterp[cptinterp['unit']!='Base_D0']
        cptinterp = cptinterp[cptinterp['unit']!='Base_C1']
        cptinterp = cptinterp[cptinterp['unit']!='Base_C12']
        # merge with cptinterp to get the interpretation and remove rows with no interpretatiion
        df_m = pd.merge(df_qc, cptinterp[['borehole', 'z', 'unit']], how='outer', on=['borehole','z']).dropna(subset=['z'])
        # Get indexes of max MD
        idx = df_m.groupby(by='borehole')['z'].transform(min) == df_m['z']
        EOCPT = df_m[idx].copy()
        BaseD0 = df_m[idx].copy()
        df_m = df_m.drop(df_m.index[idx])
        BaseD0['unit'] = 'Base_D0'
        EOCPT['unit'] = 'EOCPT'
        BaseD0['qc'] = np.nan
        BaseD0['z'] = BaseD0['Z']+0.2
        BaseD0['MD'] = BaseD0['MD']-0.2
        df_m = pd.concat([pd.concat([df_m, BaseD0], ignore_index=True), EOCPT], ignore_index=True)
        # sort on borehole and Z
        df_m = df_m.sort_values(by=['borehole', 'z'])
        # Interpolate within each 'borehole' group if interpolation does not match Z in qc
        # no extrapolation
        df_m = df_m.groupby('borehole').apply(lambda group: group.interpolate(method='linear'))
        # Replace unit NaN with proper unit names
        df_m = df_m.fillna(method='ffill')
        df_qc = df_m.copy()
    return df_qc

#%% Load CPT data - LAS format with LASIO
def load_cpts_qc_LASIO(directory, cptloc=pd.DataFrame([]), cptinterp=pd.DataFrame([])):
    # Import all LAS in folder into dataframe
    df_qc = pd.DataFrame()
    if np.ndim(directory)==0:
        directory=[directory]
    for i in range(np.shape(directory)[0]):
        for filename in os.listdir(directory[i]):
            if filename.endswith(".las" or ".LAS"):
                LAS = lasio.read(os.path.join(directory[i], filename))
                df_las = LAS.df()
                df_las.reset_index(inplace=True)
                df_las = df_las.rename(columns = {'DEPT':'MD'})
                df_las = df_las.rename(columns = {'QC':'qc'})
                df_las = df_las.rename(columns = {'FS':'fs'})
                df_las = df_las.rename(columns = {'U2':'u2'})
                df_las = df_las.rename(columns = {'QT:1':'qt'})
                df_las = df_las.rename(columns = {'QT:2':'Qt'})
                df_las = df_las.rename(columns = {'BQ':'Bq'})
                df_las = df_las.rename(columns = {'FR':'Fr'})
                df_las['borehole'] = os.path.splitext(filename)[0]
                if os.path.splitext(filename)[0] == 'TNW024-PCPT':  #############TNW#####################
                    df_las = df_las.loc[df_las['MD']<=21.5,:]
                    print('EDIIIIIIIIIIIIIIIIIIIIIITTTTTTTTTTTT')
                if not cptloc.empty:
                    print(os.path.splitext(filename)[0])
                    # print(os.path.splitext(filename)[0], (os.path.splitext(filename)[0] in cptloc.index.values) | (os.path.splitext(filename)[0] in cptloc['borehole'].values))
                else:
                    print(os.path.splitext(filename)[0])
                df_qc = df_qc.append(df_las)
                
    # Merge with cptloc if it exists and convert MD to Z
    if cptloc.empty:
        print('No CPT location defined')
    else:
        df_qc = pd.merge(cptloc, df_qc, how='inner', on=['borehole'])
        # df_qc['Z'] = df_qc['WD']-df_qc.index

    # # Merge with cptinterp if it exists
    # if cptinterp.empty:
    #     print('No CPT interpretation loaded')
    # else:
    #     #remove EOCPT and BaseD0 from interpcpt to be reassigned ############Very specific for HKZ ################
    #     cptinterp = cptinterp[cptinterp['unit']!='EOCPT']
    #     cptinterp = cptinterp[cptinterp['unit']!='Base_D0']
    #     cptinterp = cptinterp[cptinterp['unit']!='Base_C1']
    #     cptinterp = cptinterp[cptinterp['unit']!='Base_C12']
    #     # merge with cptinterp to get the interpretation and remove rows with no interpretatiion
    #     df_m = pd.merge(df_qc, cptinterp[['borehole', 'z', 'unit']], how='outer', on=['borehole','z']).dropna(subset=['z'])
    #     # Get indexes of max MD
    #     idx = df_m.groupby(by='borehole')['z'].transform(min) == df_m['z']
    #     EOCPT = df_m[idx].copy()
    #     BaseD0 = df_m[idx].copy()
    #     df_m = df_m.drop(df_m.index[idx])
    #     BaseD0['unit'] = 'Base_D0'
    #     EOCPT['unit'] = 'EOCPT'
    #     BaseD0['qc'] = np.nan
    #     BaseD0['z'] = BaseD0['Z']+0.2
    #     BaseD0['MD'] = BaseD0['MD']-0.2
    #     df_m = pd.concat([pd.concat([df_m, BaseD0], ignore_index=True), EOCPT], ignore_index=True)
    #     # sort on borehole and Z
    #     df_m = df_m.sort_values(by=['borehole', 'z'])
    #     # Interpolate within each 'borehole' group if interpolation does not match Z in qc
    #     # no extrapolation
    #     df_m = df_m.groupby('borehole').apply(lambda group: group.interpolate(method='linear'))
    #     # Replace unit NaN with proper unit names
    #     df_m = df_m.fillna(method='ffill')
    #     df_qc = df_m.copy()
    return df_qc



#%% Load CPT data NGI format
def load_cpts_NGI(directory, cptloc=pd.DataFrame(), cptinterp=pd.DataFrame()):
    df_cpts = pd.DataFrame()
    for filename in os.listdir(directory):        
        if filename.endswith(".data"):
            cpt = pd.read_csv(os.path.join(directory, filename), 
                               skiprows=[0,1,2,3,4,5,7], header=0, 
                               delim_whitespace=True, index_col=False)
            cpt['borehole'] = os.path.splitext(filename)[0]
            cpt = cpt.iloc[:-1] # remove last row
            print(os.path.splitext(filename)[0])
            df_cpts = df_cpts.append(cpt)
    # Merge with cptloc if it exists
    if cptloc.empty:
        print('No CPT location defined')
    else:
        df_cpts = pd.merge(cptloc, df_cpts, how='inner', on=['borehole'])
        print('Merge cpt locations')
    # Merge with cptinterp if it exists
    if cptinterp.empty:
        print('No CPT interpretation loaded')

    return df_cpts


#%% Load polygon
def load_polygon_csv(filepath):
    df_polygon = pd.read_csv(filepath, sep=' ', header=None, names=['x','y'])
    return df_polygon


#%%
# # %% load netCDF3 (or 1 or 2) files (horizons)
# def load_netcdf(filepath):
#     f = netcdf.netcdf_file(filepath, 'r')
#     return df_grd

    






















