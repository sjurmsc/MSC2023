# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 13:09:45 2021

@author: GuS
"""

#%% import Libraries
import numpy as np
import pandas as pd
import segyio
from scipy.interpolate import RBFInterpolator, LinearNDInterpolator
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.spatial import KDTree
import scipy.stats as stats
import os

#%% Some functions
def read_ggm_segy(path_ggm):
    src = segyio.open(path_ggm)
    ggm_grd = segyio.tools.cube(src)
    ggm_grd = ggm_grd[:,:,:]
    z = src.samples
    # z = z[::10]
    scalco = src.attributes(segyio.TraceField.SourceGroupScalar)[0]
    if scalco<0:
        x = src.attributes(segyio.TraceField.SourceX)[:]/-scalco
        y = src.attributes(segyio.TraceField.SourceY)[:]/-scalco
    elif scalco>0:
        x = src.attributes(segyio.TraceField.SourceX)[:]*scalco
        y = src.attributes(segyio.TraceField.SourceY)[:]*scalco
    else:
        x = src.attributes(segyio.TraceField.SourceX)[:]
        y = src.attributes(segyio.TraceField.SourceY)[:]
    XX = np.reshape(x, np.shape(ggm_grd)[0:2])
    YY = np.reshape(y, np.shape(ggm_grd)[0:2])
    ZZZ = np.tile(z, (np.shape(ggm_grd)[0],np.shape(ggm_grd)[1],1)).astype('float32')
    src.close()
    return ggm_grd, XX, YY, ZZZ


def read_earthgrid(path_file):
    df_tmp = pd.read_csv(path_file, sep=" ", names=['x','y','z_bsl','il','xl'], skiprows=20)
    # zz_sf = df_tmp['z_bsl'].values.reshape([1401,1401])
    return df_tmp


def zbsl2zbsf(ZZZ_bsl, ZZ_sf):
    ZZZ_sf = np.tile(ZZ_sf.T, (np.shape(ZZZ_bsl)[2],1,1)).T
    ZZZ_bsf = ZZZ_bsl-ZZZ_sf
    return ZZZ_bsf
    

def build_qcUK3D(unitlist, df_unitmap, df_UKs, ZZZ_bsf, XX, YY, ggm_grd):
    qc_grd = np.empty(np.shape(ggm_grd), dtype='float32')*np.nan
    XXX = np.tile(XX.T, (np.shape(ZZZ_bsf)[2],1,1)).T
    YYY = np.tile(YY.T, (np.shape(ZZZ_bsf)[2],1,1)).T
    for unit in unitlist: #unitlist[2:10]:
        # uid = int(unit.lstrip('GGM').rstrip('A'))
        [uid] = df_unitmap.loc[df_unitmap['unit']==unit,'uid'].values
        print(unit, uid)
        [UK3D] = df_UKs.loc[df_UKs['unit']==unit, 'RFreg_obj']                       
        print('update qc_grid')

        ZZZ_unit = ZZZ_bsf[ggm_grd==uid]
        XXX_unit = XXX[ggm_grd==uid]
        YYY_unit = YYY[ggm_grd==uid]
        
        # qc_tmp, _ = UK3D.execute('points', XXX_unit, YYY_unit, ZZZ_unit)
        
        if unit=='GGM61':
            n_split=1000
        else:
            n_split=10
        ii = 0
        qc_tmp = []
        for Z_split in np.array_split(ZZZ_unit, n_split):
            print('%d/%d' %(ii, n_split-1))
            X_split = np.array_split(XXX_unit, n_split)[ii]
            Y_split = np.array_split(YYY_unit, n_split)[ii]      
            qc_split, _ = UK3D.execute('points', X_split, Y_split, Z_split)
            qc_tmp.append(qc_split)
            ii = ii+1
        qc_tmp = np.concatenate(qc_tmp, axis=0)        
        
        qc_grd[ggm_grd==uid] = qc_tmp
        
    return qc_grd, XXX, YYY


def nptosegy(path_file, path_ggm, qc_grd):
    segyio.tools.from_array(path_file, qc_grd, format=5)
    src =  segyio.open(path_ggm)
    dst = segyio.open(path_file, 'r+')
    dst.bin = src.bin
    # dst.bin = {segyio.BinField.Samples: len(src.samples[::10])}
    # dst.samples = src.samples[::10]
    dst.header = src.header
    dst.close()
    src.close()


#%% Load UK3D predictor
print('Load UK3D predictor')
path_UKs = '../../09-Results/Stage-01/UKriging3D-Kfold-Scores.pkl'
df_UKs = pd.read_pickle(path_UKs)

df_UKs.loc[df_UKs['unit']=='GGM31C','unit'] = 'GGM31B'
df_UKs.loc[df_UKs['unit']=='GGB01C','unit'] = 'GGB01B'

df_UKs = df_UKs.loc[df_UKs['feature']=='qc',:]

#%% make unit list
print('Map units')
unitlist = df_UKs['unit'].unique()
unitlist = np.sort(unitlist)
df_unitmap = pd.DataFrame(data= {'unit': ['GGM01', 'GGM02', 'GGM03', 'GGM11', 'GGM21',
                                          'GGM22', 'GGM23', 'GGM24', 'GGM31A', 'GGM31B', 
                                          'GGM32', 'GGB01A', 'GGB01B', 'GGM33', 'GGM41', 
                                          'GGM42', 'GGM51', 'GGM52', 'GGM53', 'GGM54', 'GGM61'], 
                                 'uid': [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14,
                                         15, 16, 17, 18, 19, 20, 21, 22, 23]})
# print('write unit mapping')
# df_unitmap.to_csv("P:/2019/07/20190798/Calculations/GroundModel/09-Results/Stage-03/qc_pred_UK3D/Unit_mapping.csv", index=False)

#%% Load GGM
print('Read GGM segy')
path_ggm = "P:/2019/07/20190798/Calculations/GroundModel/09-Results/Stage-03/00-StructuralModel/StructuralModel.sgy"
ggm_grd, XX, YY, ZZZ = read_ggm_segy(path_ggm)

#%% Load seafloor
print('Load seafloor')
# path_sf = "D:/TNW/Grids/R00-25_25.dat"
# df_sf = read_earthgrid(path_sf)
path_sf = "P:/2019/07/20190798/Background/SAND_Geophysics/2021-08-26/01_Picks/SAND_R00_Z1_XY.dat"
df_sf = pd.read_csv(path_sf, sep='\s+', names=['x','y', 'twt', 'z_bsl'])

interp = LinearNDInterpolator(list(zip(df_sf.x, df_sf.y)), df_sf.z_bsl)
ZZ_sf = interp(XX.flatten(), YY.flatten())
ZZ_sf = ZZ_sf.reshape(np.shape(XX))

del df_sf, interp

#%% z_bsl to z_bsf
print('z_bsl to z_bsf')
ZZZ_bsf = zbsl2zbsf(ZZZ, ZZ_sf)


#%% Load horizons and variance
dir_path_hor = 'P:/2019/07/20190798/Calculations/GroundModel/01-Data/Interp_Seismic/Picks/'
df_horizons = pd.DataFrame([])
df_horizons_var = pd.DataFrame([])
for filename in os.listdir(dir_path_hor):
    if filename.endswith("outside.txt"):
        hor_path = os.path.join(dir_path_hor, filename)
        unit = filename[11:14]
        print(filename)
        df_tmp = pd.read_csv(hor_path, sep=" ", names=['x','y','z_bsl','il','xl'], skiprows=20)
        df_tmp['unit'] = unit
        df_horizons = pd.concat([df_horizons,df_tmp[['x','y','z_bsl','unit']]])
    elif filename.endswith("_var.txt"):
        var_path = os.path.join(dir_path_hor, filename)
        unit = filename[11:14]
        print(filename)
        df_tmp_var = pd.read_csv(var_path, sep=" ", names=['x','y','var','il','xl'], skiprows=20)
        df_tmp_var['unit'] = unit
        df_horizons_var = pd.concat([df_horizons_var,df_tmp_var[['x','y','var','unit']]])


#%% Quadtree decomposition
hor_list = df_horizons['unit'].unique()
df_kdt = pd.DataFrame()
for hor in hor_list:
    print(hor)
    df_hor = df_horizons.loc[df_horizons['unit']==hor,:]
    df_hor_var = df_horizons_var.loc[df_horizons_var['unit']==hor,:]
    tree = KDTree(np.c_[df_hor['x'], df_hor['y']])
    df_tmp = pd.DataFrame(data={'tree': [tree], 'unit': [hor], 'hor': [df_hor], 'var': [df_hor_var]})
    df_kdt = pd.concat([df_kdt, df_tmp])
    

#%%
unc_grd = ggm_grd*0
for ii in range(0, np.shape(ZZ_sf)[0]):
    print(str(ii+1) + '/' + str(np.shape(ZZ_sf)[0]))
    for jj in range(0, np.shape(ZZ_sf)[1]):
        if ~np.isnan(ZZ_sf[ii,jj]):
            z_sf = ZZZ[ii,jj,:]
            for hor in hor_list:
                [tree] = df_kdt.loc[df_kdt['unit']==hor,'tree']
                x = XX[ii,jj]
                y = YY[ii,jj]
                kdd, kdi = tree.query([x, y], k=1)
                if kdd<25:
                    [df_hor] = df_kdt.loc[df_kdt['unit']==hor,'hor']
                    [df_hor_var] = df_kdt.loc[df_kdt['unit']==hor,'var']
                    if hor in ['R60', 'R53', 'R52']:
                        var = df_hor_var.iloc[kdi, 2]/4
                    else:
                        var = df_hor_var.iloc[kdi, 2]
                    z_var = df_hor.iloc[kdi, 2]
                    sigma = np.sqrt(var)
                    confint = z_sf[(z_sf>=z_var-2.5*sigma) & (z_sf<=z_var+2.5*sigma)]
                    if len(confint)>0:
                        nd = stats.norm.pdf(confint, z_var, sigma)
                        nd_norm = nd/np.max(nd)
                        nd_var = np.maximum(unc_grd[ii,jj,(z_sf>=z_var-2.5*sigma) & (z_sf<=z_var+2.5*sigma)], nd_norm*var)
                        unc_grd[ii,jj,(z_sf>=z_var-2.5*sigma) & (z_sf<=z_var+2.5*sigma)] = nd_var
                    

#%% Write SEGY
print('Write qc_grid to segy...')
path_file = "P:/2019/07/20190798/Calculations/GroundModel/09-Results/Stage-03/00-StructuralModel/StrucutralModel_uncertainty.sgy"
nptosegy(path_file, path_ggm, unc_grd)
print('done')


    










