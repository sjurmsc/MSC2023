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
    
def build_slopeintercept_map(slope_maps, intercept_maps, var_maps, unitlist, df_LL, XX, YY):
    ii = 0
    for unit in unitlist:
        print(unit)
        [UK_slope] = df_LL.loc[(df_LL['unit']==unit), 'slope_kri_obj']
        [UK_intercept] = df_LL.loc[(df_LL['unit']==unit), 'intercept_kri_obj']
        slope, slope_var = UK_slope.execute('points', XX.flatten(), YY.flatten())
        slope_maps[ii,:,:] = slope.reshape(np.shape(XX))
        intercept, intercept_var = UK_intercept.execute('points', XX.flatten(), YY.flatten())
        intercept_maps[ii,:,:] = intercept.reshape(np.shape(XX))
        var = slope_var+intercept_var
        var_maps[ii,:,:] = var.reshape(np.shape(XX))
        ii = ii+1
    return slope_maps, intercept_maps, var_maps

def build_slopeintercept_grd(slope_grd, intercept_grd, var_grd, slope_maps, intercept_maps, var_maps, ggm_grd, unitlist, df_unitmap):
    ii = 0
    for unit in unitlist:
        [uid] = df_unitmap.loc[df_unitmap['unit']==unit, 'uid'].values
        print(unit, uid)
        print('update slope_grd')
        slope = np.tile(slope_maps[ii,:,:].T, (np.shape(ggm_grd)[2],1,1)).T
        slope_grd[ggm_grd==uid] = slope[ggm_grd==uid]
        print('update intercept_grd')
        intercept = np.tile(intercept_maps[ii,:,:].T, (np.shape(ggm_grd)[2],1,1)).T
        intercept_grd[ggm_grd==uid] = intercept[ggm_grd==uid]
        print('update var_grd')
        var = np.tile(var_maps[ii,:,:].T, (np.shape(ggm_grd)[2],1,1)).T
        var_grd[ggm_grd==uid] = var[ggm_grd==uid]
        ii = ii+1
    return slope_grd, intercept_grd, var_grd


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


#%% Load LL predictor
print('Load LL predictor')
path_LLs = '../../09-Results/Stage-01/LocalLinearFitwBounds-UKriging-GroupKfold-Scores.pkl'
df_LLs = pd.read_pickle(path_LLs)

df_LLs.loc[df_LLs['unit']=='GGM31C','unit'] = 'GGM31B'
df_LLs.loc[df_LLs['unit']=='GGB01C','unit'] = 'GGB01B'

df_LL = df_LLs.loc[df_LLs['feature']=='qc',:]
# df_LL = df_LLs.loc[df_LLs['feature']=='fs',:]
# df_LL = df_LLs.loc[df_LLs['feature']=='u2',:]

#%% make unit list
print('Map units')
unitlist = df_LL['unit'].unique()
unitlist = np.sort(unitlist)
df_unitmap = pd.DataFrame(data= {'unit': ['GGM01', 'GGM02', 'GGM03', 'GGM11', 'GGM21',
                                          'GGM22', 'GGM23', 'GGM24', 'GGM31A', 'GGM31B', 
                                          'GGM32', 'GGB01A', 'GGB01B', 'GGM33', 'GGM41', 
                                          'GGM42', 'GGM51', 'GGM52', 'GGM53', 'GGM54', 'GGM61'], 
                                 'uid': [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14,
                                         15, 16, 17, 18, 19, 20, 21, 22, 23]})
print('write unit mapping')
df_unitmap.to_csv("P:/2019/07/20190798/Calculations/GroundModel/09-Results/Stage-03/Unit_mapping.csv", index=False)

#%% Load GGM
print('Read GGM segy')
path_ggm = "P:/2019/07/20190798/Calculations/GroundModel/09-Results/Stage-03/00-StructuralModel/StructuralModel.sgy"
ggm_grd, XX, YY, ZZZ = read_ggm_segy(path_ggm)


#%% Load seafloor
print('Load seafloor')
path_sf = "D:/TNW/Grids/R00-25_25.dat"
df_sf = read_earthgrid(path_sf)
path_sf = "P:/2019/07/20190798/Background/SAND_Geophysics/2021-08-26/01_Picks/SAND_R00_Z1_XY.dat"
df_sf = pd.read_csv(path_sf, sep='\s+', names=['x','y', 'twt', 'z_bsl'])

interp = LinearNDInterpolator(list(zip(df_sf.x, df_sf.y)), df_sf.z_bsl)
ZZ_sf = interp(XX.flatten(), YY.flatten())
ZZ_sf = ZZ_sf.reshape(np.shape(XX))

del df_sf, interp

#%% z_bsl to z_bsf
print('z_bsl to z_bsf')
ZZZ_bsf = zbsl2zbsf(ZZZ, ZZ_sf)

#%% Build slope and intercept maps
print('Build Slope and intercept 2D grids')
slope_maps = np.empty([len(unitlist), np.shape(ggm_grd)[0], np.shape(ggm_grd)[1]], dtype='float32')*np.nan
intercept_maps = np.empty(np.shape(slope_maps), dtype='float32')*np.nan
var_maps = np.empty(np.shape(slope_maps), dtype='float32')*np.nan
slope_maps, intercept_maps, var_maps = build_slopeintercept_map(slope_maps, intercept_maps, var_maps, unitlist, df_LL, XX, YY)

#%% Build slope and intercept 3D grids
print('Build slope and intercept 3D grids')
slope_grd = np.empty(np.shape(ggm_grd), dtype='float32')*np.nan
intercept_grd = np.empty(np.shape(ggm_grd), dtype='float32')*np.nan
var_grd = np.empty(np.shape(ggm_grd), dtype='float32')*np.nan
slope_grd, intercept_grd, var_grd = build_slopeintercept_grd(slope_grd, intercept_grd, var_grd, 
                                                             slope_maps, intercept_maps, var_maps, 
                                                             ggm_grd, unitlist, df_unitmap)
# del ggm_grd

#%% Reconstruct CPT parameter 3D grid
print('Build qc_pred 3D grid')
qc_grd = slope_grd*ZZZ_bsf + intercept_grd


#%% Build low and high estimate
print('Build low and high')
qc_low_grd = np.empty(np.shape(ggm_grd), dtype='float32')*np.nan
qc_high_grd = np.empty(np.shape(ggm_grd), dtype='float32')*np.nan
qc_low_grd = qc_grd - np.sqrt(var_grd)
qc_low_grd[qc_low_grd<=0]=0
qc_high_grd = qc_grd + np.sqrt(var_grd)

    
#%% Write SEGY
# print('Write qc_grid to segy...')
qc_grd_segy = qc_grd.copy()
qc_grd_segy[qc_grd_segy<=-1]=-1
qc_grd_segy[np.isnan(qc_grd_segy)]=-1

path_file = "P:/2019/07/20190798/Calculations/GroundModel/09-Results/Stage-03/01-LocalLinear/LL_qc.sgy"
nptosegy(path_file, path_ggm, qc_grd)
print('done')
# np2netcdf(path_file, XX, YY, ZZZ[0,0,:], qc_grd, 'UK_qc')

print('Write qc_low_grid to segy...')
path_file = "P:/2019/07/20190798/Calculations/GroundModel/09-Results/Stage-03/01-LocalLinear/LL_qc_low.sgy"
nptosegy(path_file, path_ggm, qc_low_grd)
print('done')

print('Write qc_high_grid to segy...')
path_file = "P:/2019/07/20190798/Calculations/GroundModel/09-Results/Stage-03/01-LocalLinear/LL_qc_high.sgy"
nptosegy(path_file, path_ggm, qc_high_grd)
print('done')



# path_file = "P:/2019/07/20190798/Calculations/GroundModel/09-Results/Stage-03/01-LocalLinear/LL_fs.sgy"
# nptosegy(path_file, path_ggm, qc_grd)
# print('done')
# # np2netcdf(path_file, XX, YY, ZZZ[0,0,:], qc_grd, 'UK_qc')

# print('Write qc_low_grid to segy...')
# path_file = "P:/2019/07/20190798/Calculations/GroundModel/09-Results/Stage-03/01-LocalLinear/LL_fs_low.sgy"
# nptosegy(path_file, path_ggm, qc_low_grd)
# print('done')

# print('Write qc_high_grid to segy...')
# path_file = "P:/2019/07/20190798/Calculations/GroundModel/09-Results/Stage-03/01-LocalLinear/LL_fs_high.sgy"
# nptosegy(path_file, path_ggm, qc_high_grd)
# print('done')

# print('Write qc_var_grid to segy...')
# path_file = "P:/2019/07/20190798/Calculations/GroundModel/09-Results/Stage-03/01-LocalLinear/LL_fs_var.sgy"
# nptosegy(path_file, path_ggm, var_grd)
# print('done')

    

# path_file = "P:/2019/07/20190798/Calculations/GroundModel/09-Results/Stage-03/01-LocalLinear/LL-u2_best.sgy"
# nptosegy(path_file, path_ggm, qc_grd)
# print('done')
# # np2netcdf(path_file, XX, YY, ZZZ[0,0,:], qc_grd, 'UK_qc')

# print('Write qc_low_grid to segy...')
# path_file = "P:/2019/07/20190798/Calculations/GroundModel/09-Results/Stage-03/01-LocalLinear/LL-u2_lowmean.sgy"
# nptosegy(path_file, path_ggm, qc_low_grd)
# print('done')

# print('Write qc_high_grid to segy...')
# path_file = "P:/2019/07/20190798/Calculations/GroundModel/09-Results/Stage-03/01-LocalLinear/LL-u2_highmean.sgy"
# nptosegy(path_file, path_ggm, qc_high_grd)
# print('done')

# print('Write qc_var_grid to segy...')
# path_file = "P:/2019/07/20190798/Calculations/GroundModel/09-Results/Stage-03/01-LocalLinear/LL-u2_var.sgy"
# nptosegy(path_file, path_ggm, var_grd)
# print('done')










