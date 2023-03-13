# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 10:42:25 2021

@author: GuS
modif by JRD

2021-04-07
- buildstrati
add of a flag value for seismic stratigraphy or geotech stratigraphy

"""
#%% Import libraries
import os
import segyio
import lasio
import netCDF4 as nc
import pandas as pd
import numpy as np
import scipy.spatial as spatial
from scipy.interpolate import interp1d

import GM_LoadData as GML

#%% Define functions
def t2d_Vint(dt, t, Vp, Vp_water=1500):
    z = np.empty(np.shape(t))
    for ii in range(0, len(t)):
        if ii==1:
            z[ii] = t[0]*Vp_water
        else:
            z[ii] = z[ii-1]+Vp[ii]*dt/2

def load_NAV(dirpath_NAV):
    # Load 2D UHRS navigation
    print('Load navigation')
    df_NAV = pd.DataFrame()
    for filename in os.listdir(dirpath_NAV):
        if filename.endswith(".nav"):
            # print(filename[:-4])
            xyNAV = pd.read_csv(os.path.join(dirpath_NAV, filename), sep='\s+', header=None, names=['x', 'y', 'tracl', 'ep'])
            xyNAV['line'] = filename[:-4]
            df_NAV = df_NAV.append(xyNAV)
    print('Navigation loaded')
    return df_NAV

def get_seis_at_CPT(df_NAV, dirpath_sgy, df_CPT_loc=pd.DataFrame([]), n_tr = 1):
    ''' 
    Extract seismic trace(s) closest to CPT location and add to database
    df_NAV: dataframe with Navigation data
    dirpath_sgy: path to directory with corresponding SEGY files
    df_CPT_loc: dataframe with CPT locations
    n_tr: number of traces to extract
    '''
    # Quadtree decomposition of the Navigation
    xy = df_NAV[['x','y']]
    print('\nQuadtree decomposition ...\n')
    tree = spatial.KDTree(xy)
        
    # Loop over all CPT locations find closest location and add trace to database
    print('Merging seismic traces')
    df_seisall = pd.DataFrame()
    for ind, row in df_CPT_loc.iterrows():
        distance, index = tree.query(row[['x','y']], k=n_tr)
        if n_tr>1:
            indexes = index.flatten()
            distances = distance.flatten()
            linenames = df_NAV.iloc[indexes, df_NAV.columns.get_loc('line')]
            tracls = df_NAV.iloc[indexes, df_NAV.columns.get_loc('tracl')]
        else:
            indexes = index
            distances = [np.array(distance)]
            linenames = [df_NAV.iloc[indexes, df_NAV.columns.get_loc('line')]]
            tracls = [df_NAV.iloc[indexes, df_NAV.columns.get_loc('tracl')]]
           
        # Extract seismic trace using segyio
        ii = 0
        df_seisloc = pd.DataFrame()
        for line, tracl, dist in zip(linenames, tracls, distances):
            ii = ii+1
            path_seis = dirpath_sgy + line + '.sgy'
            with segyio.open(path_seis, 'r', ignore_geometry=True) as f:
                # Get basic attributes
                sample_rate = segyio.tools.dt(f) # in mm
                n_samples = f.samples.size
                z_seis = f.samples*1000        # in mm
                # Load nearest trace(s)
                tr = f.trace[tracl]
            
            # Resample trace to CPT dz: 20 mm
            dz = 20   # CPT z sampling (20 mm)
            z_bsl = np.arange(0, (n_samples-1)*sample_rate+dz, dz)
            f = interp1d(z_seis, tr, kind='slinear')
            tr_resamp = f(z_bsl)
            tr_colname = 'amp_tr_' + str(ii)
            dist_colname = 'dist_tr_' + str(ii)
            df_seis = pd.DataFrame({'z_bsl': z_bsl, tr_colname: tr_resamp})
            df_seis[dist_colname] = dist
            df_seis['borehole'] = row['borehole']
            df_seis['x'] = row['x']
            df_seis['y'] = row['y']
            df_seis['WD'] = row['WD']
            df_seis['ID'] = row['ID']
            # df_seis.set_index('borehole', inplace=True)
            df_seisloc = pd.concat((df_seisloc, df_seis), axis=1)
            df_seisloc = df_seisloc.loc[:,~df_seisloc.columns.duplicated()]
        df_seisall = df_seisall.append(df_seisloc)
        df_seisall['z_bsl'] = df_seisall['z_bsl'].astype(int)
    print('seismic traces merged')
    return df_seisall

def get_seisAI_at_CPT(df_NAV, df_database, dirpath_AI_sgy, df_CPT_loc=pd.DataFrame([]), n_tr = 1):
    ''' 
    Extract seismic trace(s) closest to CPT location and add to database
    df_NAV: dataframe with Navigation data
    dirpath_sgy: path to directory with corresponding SEGY files
    df_CPT_loc: dataframe with CPT locations
    n_tr: number of traces to extract
    '''
    # Quadtree decomposition of the Navigation
    xy = df_NAV[['x','y']]
    print('\nQuadtree decomposition ...\n')
    tree = spatial.KDTree(xy)
        
    # Loop over all CPT locations find closest location and add trace to database
    print('Merging seismic traces')
    for ind, row in df_CPT_loc.iterrows():
        distance, index = tree.query(row[['x','y']], k=n_tr)
        if n_tr>1:
            indexes = index.flatten()
            distances = distance.flatten()
            linenames = df_NAV.iloc[indexes, df_NAV.columns.get_loc('line')]
            tracls = df_NAV.iloc[indexes, df_NAV.columns.get_loc('tracl')]
        else:
            indexes = index
            distances = [np.array(distance)]
            linenames = [df_NAV.iloc[indexes, df_NAV.columns.get_loc('line')]]
            tracls = [df_NAV.iloc[indexes, df_NAV.columns.get_loc('tracl')]]
           
        # Extract seismic trace using segyio
        ii = 0
        df_seisloc = pd.DataFrame()
        for line, tracl, dist in zip(linenames, tracls, distances):
            ii = ii+1
            path_seis_AI = dirpath_AI_sgy + line[:-4] + '.Abs_Zp.sgy'
            print(path_seis_AI)
            with segyio.open(path_seis_AI, 'r', ignore_geometry=True) as f:
                # Get basic attributes
                sample_rate = segyio.tools.dt(f) # in mm
                n_samples = f.samples.size
                t_AI = f.samples
                z_AI = t2d_Vint(sample_rate, t_AI, Vp, Vp_water=1500)
                # Load nearest trace(s)
                tr_AI = f.trace[tracl]
                    
            # Resample trace to CPT dz: 20 mm
            dz = 20   # CPT z sampling (20 mm)
            z_bsl = np.arange(0, (n_samples-1)*sample_rate+dz, dz)
            f = interp1d(z_AI, tr_AI, kind='slinear')
            tr_AI_resamp = f(z_bsl)
            tr_ai_colname = 'AI_tr_' + str(ii)
            df_seis = pd.DataFrame({'z_bsl': z_bsl, tr_ai_colname: tr_AI_resamp})
            df_seis['ID'] = row['ID']
            # df_seis.set_index('borehole', inplace=True)
            df_seisloc = pd.concat((df_seisloc, df_seis), axis=1)
            df_seisloc = df_seisloc.loc[:,~df_seisloc.columns.duplicated()]
            
        # Merge with database
        df_database_m = pd.merge(df_database, df_seisloc[['ID','z_bsl', attr]], how='left', on=['ID', 'z_bsl'])
    print('seismic traces merged')
    return df_database_m


def buildstrati(path_strati, df_CPT_loc, df_database, typestrati='seis'):
    '''
    Parameters
    ----------
    path_strati : str, filepath
        File path to the xls spreadsheet containing the unit definition.
    df_CPT_loc : dataframe, optional
        Dataframe containing the CPT location.
    df_database : dataframe, optional
        Dataframe with the database to add the strati to.
    typestrati : 'seis' or 'geo'
        To adapt the column name to the loaded strati.

    Returns
    -------
    df_database : dataframe
        Dataframe of the database augmented with the stratigraphy
    '''
    
    print('\nBuilding Strati \n')
    #read Spreadsheet with unit top/bottom
    df_strati = pd.read_excel(path_strati)
    # unnamed=sorted(list(df_strati))[-1]
    # df_strati = df_strati.drop([unnamed], axis=1)
    
    # Build strati sequence for each location and merge with database
    for ID in df_CPT_loc['ID'].unique():
        print(int(ID))
        df_strati_loc = df_strati.loc[df_strati['CPT']==ID, :].dropna(axis=1)
        if not df_strati_loc.empty:
            units_loc = np.unique([string[:-2] for string in list(df_strati_loc)[1:]])
            for unit in units_loc:
                print(unit)
                unitT = 1000*df_strati_loc[unit + ' T'].values[0]
                unitB = 1000*df_strati_loc[unit + ' B'].values[0]
                if typestrati=='seis':
                    df_database.loc[(df_database['ID']==ID) & (df_database['z_bsf']>=unitT) & (df_database['z_bsf']<unitB), 'unit'] = unit
                elif typestrati=='geo':
                    df_database.loc[(df_database['ID']==ID) & (df_database['z_bsf']>=unitT) & (df_database['z_bsf']<unitB), 'unit_geo'] = unit
    return df_strati, df_database



def load_attr(dirpath, df_database=pd.DataFrame([]), attr='undef'):
    '''
    Function to load Qp (or Vp or AI) files and merge with database
    Database must have ['ID', 'z_bsf']
    
    Parameters
    ----------
    dirpath : directory path
        Directory path containing Qp files
    df_database : dataframe, optional
        Dataframe containing the database to merge with. The default is pd.DataFrame([]).
    attr : str, optional
        String specifying the attribute type. The default is 'undef'.

    Returns
    -------
    df_database_m : dataframe
        Dataframe of the augmented database with attribute.

    '''
    df_attr = pd.DataFrame()
    for filename in os.listdir(dirpath):
        if filename.endswith(".layered.asc"):
            print(filename)
            df_attr_loc = pd.read_csv(os.path.join(dirpath, filename), 
                                      sep='\s+', header=None,
                                      names=[attr, 'TWT', 'z_bsf'], skiprows=[0])
            if not df_attr_loc.empty:
                #Resample to CPT dz
                dz = 20   # CPT z sampling (20 mm)
                z_bsf = np.arange(1000*round(df_attr_loc['z_bsf'].min()*50)/50, 1000*round(df_attr_loc['z_bsf'].max()*50)/50, dz)
                f = interp1d(1000*df_attr_loc['z_bsf'], df_attr_loc[attr], kind='nearest', bounds_error=False, fill_value=np.nan)
                attr_resamp = f(z_bsf)
                df_tmp = pd.DataFrame({'z_bsf': z_bsf, attr: attr_resamp})
                df_tmp['ID'] = int(''.join(filter(lambda i: i.isdigit(), filename)))
                df_attr = df_attr.append(df_tmp)
    # Merge with database
    df_database_m = pd.merge(df_database, df_attr[['ID','z_bsf', attr]], how='left', on=['ID', 'z_bsf'])
    return df_database_m

def load_AI_sgy(dirpath, df_database=pd.DataFrame([]), attr='undef'):
    '''
    Function to load Qp (or Vp or AI) files and merge with database
    Database must have ['ID', 'z_bsf']
    
    Parameters
    ----------
    dirpath : directory path
        Directory path containing Qp files
    df_database : dataframe, optional
        Dataframe containing the database to merge with. The default is pd.DataFrame([]).
    attr : str, optional
        String specifying the attribute type. The default is 'undef'.

    Returns
    -------
    df_database_m : dataframe
        Dataframe of the augmented database with attribute.

    '''
    df_attr = pd.DataFrame()
    for filename in os.listdir(dirpath):
        if filename.endswith(".layered.asc"):
            print(filename)
            df_attr_loc = pd.read_csv(os.path.join(dirpath, filename), 
                                      sep='\s+', header=None,
                                      names=[attr, 'TWT', 'z_bsf'], skiprows=[0])
            if not df_attr_loc.empty:
                #Resample to CPT dz
                dz = 20   # CPT z sampling (20 mm)
                z_bsf = np.arange(1000*round(df_attr_loc['z_bsf'].min()*50)/50, 1000*round(df_attr_loc['z_bsf'].max()*50)/50, dz)
                f = interp1d(1000*df_attr_loc['z_bsf'], df_attr_loc[attr], kind='nearest', bounds_error=False, fill_value=np.nan)
                attr_resamp = f(z_bsf)
                df_tmp = pd.DataFrame({'z_bsf': z_bsf, attr: attr_resamp})
                df_tmp['ID'] = int(''.join(filter(lambda i: i.isdigit(), filename)))
                df_attr = df_attr.append(df_tmp)
    # Merge with database
    df_database_m = pd.merge(df_database, df_attr[['ID','z_bsf', attr]], how='left', on=['ID', 'z_bsf'])
    return df_database_m


def get_bathy_at_CPT(df_CPT_loc):
    # Load Seafloor
    filepath = "P:/2019/07/20190798/Calculations/Bathymetry/MMT_RVO_TNW_EM2040D_5m_alldata.grd"
    fh = nc.Dataset(filepath, mode='r')
    x = fh.variables['x'][:].filled()
    y = fh.variables['y'][:].filled()
    z = fh.variables['z'][:].filled()
    fh.close()
    del filepath, fh

    # Format data for quadtree decomposition
    xx, yy = np.meshgrid(np.float32(x), np.float32(y))
    xy = np.c_[xx.ravel(), yy.ravel()]
    zz = np.c_[z.ravel()]

    # quadtree decomposition of the bathy
    tree = spatial.KDTree(xy)

    # loop over all CPT locations find closest location
    for cpt, row in df_CPT_loc.iterrows():
        # print('cpt: ' + str(cpt))
        distance, index = tree.query(row[['x','y']])
        df_CPT_loc.at[cpt, 'WD'] = -zz[index]    
    return df_CPT_loc



#%% load CPT locations from LAS files
print('Load CPT locations')
df_CPT_loc=pd.DataFrame()


for path_las_files in ['P:/2019/07/20190798/Calculations/LasFiles/LAS_from_AGS_20210518', 'P:/2019/07/20190798/Calculations/LasFiles/LAS_from_AGS_Tennet_20210518']:
    for filename in os.listdir(path_las_files):
        if filename.endswith(".las" or ".LAS"):
            LAS = lasio.read(os.path.join(path_las_files, filename))
            tmpdict={'borehole':filename[:-4], 
                     'x':float(LAS.header['Well']['LOCX'].descr), 
                     'y':float(LAS.header['Well']['LOCY'].descr),
                     'TotalDepth':np.ceil(float(LAS.header['Well']['STOP'].value))
                     }
            df_CPT_loc=df_CPT_loc.append(tmpdict, ignore_index=True)


print('CPT locations loaded\n')


print('Load Water Depth')
df_CPT_loc=get_bathy_at_CPT(df_CPT_loc)
print('Water Depth loaded\n')

# Round WaterDepth to 0.02 m (CPT dz) to match dz from resampled seismic data and convert to mm
df_CPT_loc['WD'] = (1000*round(df_CPT_loc['WD']*50)/50).astype(int)


print('Assign CPT unique location ID')      ####### SPECIFIC TNW ##########
for index, row in df_CPT_loc.iterrows(): 
    if 'TT' in row['borehole']:
        ID = 85 + int(''.join(filter(lambda i: i.isdigit(), row['borehole'])))
    else:
        ID = int(''.join(filter(lambda i: i.isdigit(), row['borehole'])))
    df_CPT_loc.loc[index, 'ID'] = ID
print('CPT unique location ID assigned \n')

del index, row, ID, tmpdict, LAS, path_las_files, filename

df_CPT_loc=df_CPT_loc[['borehole','x','y','WD','TotalDepth','ID']]


#%% Extract seismic trace(s) at CPT locations
dirpath_NAV = 'P:/2019/07/20190798/Background/2DUHRS_06_MIG_DEPTH/nav_2DUHRS_06_MIG_DPT/'
df_NAV = load_NAV(dirpath_NAV)
#%%
dirpath_sgy = 'P:/2019/07/20190798/Background/2DUHRS_06_MIG_DEPTH/'
dirpath_AI_sgy = 'P:/2019/07/20190798/Background/SAND_Geophysics/2022-06-09/00_AI/'
df_seis = get_seisAI_at_CPT(df_NAV, dirpath_sgy, dirpath_AI_sgy, df_CPT_loc, n_tr=1)

#%%
del dirpath_NAV, dirpath_sgy


#%% Load CPT data and merge with database
dirpath_CPT_data = ['P:/2019/07/20190798/Calculations/LasFiles/LAS_from_AGS_20210518', 'P:/2019/07/20190798/Calculations/LasFiles/LAS_from_AGS_Tennet_20210518']
df_CPT_data = GML.load_cpts_qc_LASIO(dirpath_CPT_data, cptloc=df_CPT_loc)
df_CPT_data['z_bsl'] = (1*df_CPT_data['WD'] + 1000*df_CPT_data['MD']).astype(int)     # in mm


# Merge with database
df_merge = pd.merge(df_CPT_data[['borehole', 'z_bsl', 'qc', 'fs', 'u2', 'qt', 'Qt', 'Bq', 'Fr', 'ICN']],
                    df_seis, how='right', on=['borehole', 'z_bsl'])
df_merge['z_bsf'] = df_merge['z_bsl']-df_merge['WD']

del dirpath_CPT_data

#%% Load seismic stratigraphy and build strati sequence for each CPT location
path_strati_seis = '../../01-Data/CPT_BH_StratFacies_20210322-RTK.xlsx'
df_strati_seis, df_database = buildstrati(path_strati=path_strati_seis,df_CPT_loc=df_CPT_loc, 
                                          df_database=df_merge,typestrati='seis')        

#%% Load geotech stratigraphy and build strati sequence for each CPT location
path_strati_geo = '../../01-Data/CPT_BH_StratFacies_20210322-GEO.xlsx'
df_strati_geo, df_database = buildstrati(path_strati=path_strati_geo,df_CPT_loc=df_CPT_loc, 
                                         df_database=df_merge,typestrati='geo')        

#%% Load Qp and merge with database
dirpath_Qp = '../../01-Data/Qlayered_20210329/'
df_database_Qp = load_attr(dirpath_Qp, df_database, attr='Qp')

#%% Load Vp and merge with database
dirpath_Vp = '../../01-Data/Vlayered_20210327/'
df_database_m = load_attr(dirpath_Vp, df_database_Qp, attr='Vp')


#%% Calculate I_SBT Robertson
# df_database_m['I_SBT'], nall = I_SBT_Robertson(df_database_m['z_bsf'], df_database_m['qt'], df_database_m['fs'], df_database_m['u2'])

#%% Add 1D seismic attributes
import GM_SeismicAttributes as SA

# Envelop   
df_database_m['envelop']=''
for bh in df_database_m['borehole'].unique():
    seis=df_database_m[df_database_m['borehole']==bh]['amp_tr_1'].values.reshape(1,-1)
    attr=SA.apply_attribute(seis,'envelop').reshape(-1)
    df_database_m.loc[df_database_m['borehole']==bh,'envelop']=attr
    
# Energy
df_database_m['energy']=''
for bh in df_database_m['borehole'].unique():
    seis=df_database_m[df_database_m['borehole']==bh]['amp_tr_1'].values.reshape(1,-1)
    attr=SA.apply_attribute(seis,'energy',win=150).reshape(-1)
    df_database_m.loc[df_database_m['borehole']==bh,'energy']=attr

# Cumulative Envelop
df_database_m['cumulative_envelop']=''
for bh in df_database_m['borehole'].unique():
    env=df_database_m[df_database_m['borehole']==bh]['envelop'].values.reshape(1,-1)
    attr=SA.apply_attribute(env,'cumulative').reshape(-1)
    df_database_m.loc[df_database_m['borehole']==bh,'cumulative_envelop']=attr
    
# Cumulative Energy    
df_database_m['cumulative_energy']=''
for bh in df_database_m['borehole'].unique():
    energy=df_database_m[df_database_m['borehole']==bh]['energy'].values.reshape(1,-1)
    attr=SA.apply_attribute(energy,'cumulative').reshape(-1)
    df_database_m.loc[df_database_m['borehole']==bh,'cumulative_energy']=attr
    
# env     = apply_attribute(seis,'envelop')
# cumenv  = apply_attribute(env,'cumulative')
# ene    = apply_attribute(seis,'energy',win=20)
# cumene = apply_attribute(ene,'cumulative')

#%% Add K-fold groups from MEV
# path_grp = '../../01-Data/borehole.groups' 
# df_grp = pd.read_csv(path_grp, sep='\s+', header=None, names=['ID', 'kfold'])
# df_database_m = pd.merge(df_database_m, df_grp, how='left', on='ID')

#%% Final touch
# limit to -2000<z_bsl<Z_CPT_max [mm]
df_database_m = df_database_m.loc[(df_database_m['z_bsf']>=-2000) & (df_database_m['z_bsl']<=df_CPT_data['z_bsl'].max()), :]
# fall back to metre
df_database_m.loc[:,['WD', 'z_bsl', 'z_bsf']] = df_database_m.loc[:,['WD', 'z_bsl', 'z_bsf']]/1000
# Pad Qp and Vp with last value
df_database_m['Qp'] = df_database_m['Qp'].fillna(method='ffill')
df_database_m['Vp'] = df_database_m['Vp'].fillna(method='ffill')

#%% Export database to csv
path_datacsv = '../../09-Results/Stage-01/Database.csv'
print('Writing to csv ...')
df_database_m.to_csv(path_datacsv, index=False)
print('csv file written: ', path_datacsv)

path_datapkl = '../../09-Results/Stage-01/Database.pkl'
print('Writing to pickle ...')
df_database_m.to_pickle(path_datapkl)
print('pickle file written: ', path_datapkl)














