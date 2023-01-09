# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 15:26:21 2022

@author: JRD
"""

from opening_segy import *
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors

tf.keras.backend.clear_session()

# LOADING FILENAMES
# path_seis=os.path.join('P:','2019','07','20190798','Background','2DUHRS_06_MIG_DEPTH')
path=os.path.join('C:\\','Users','JRD','OneDrive - NGI','Python project','TCN-Marmousi','TNW data')


file=[]
for ff in os.listdir(path):
    if ('BX06' in ff) and (not '_not_' in ff):
        file.append(ff)

# LOADING DATA
df1,z1=opening_segy(os.path.join(path,file[0]))
df2,z2=opening_segy(os.path.join(path,file[1]))
df3,z3=opening_segy(os.path.join(path,file[2]))

text=load_segy_textual_header(os.path.join(path,file[0]))


data1=np.stack(df1.trace.to_numpy())
data2=np.stack(df2.trace.to_numpy())
data3=np.stack(df3.trace.to_numpy())

print(file[0],np.shape(data1),'  dz:',str(z1[1]-z1[0]))
print(file[1],np.shape(data2),'  dz:',str(z2[1]-z2[0]))
print(file[2],np.shape(data3),'  dz:',str(z3[1]-z3[0]))



def ai_to_reflectivity(ai,win=7,threshold=8e-4):
    '''
    Acoustic Impedance to Reflectivity
    '''    
    # compute reflectivity coeff
    refl=np.zeros(np.shape(ai))
    for i in range(len(ai)-1):
        R=(ai[i+1]-ai[i])/(ai[i+1]+ai[i])
        refl[i+1]=R
      
    ind=[]
    for i in range(win,len(ai)-win):
        for kk in range(1,win):
            if (np.abs(refl[i]-refl[i-kk])<threshold) or (np.abs(refl[i]-refl[i+kk])<threshold):
                ind.append(i)
                break     
    refl[ind]=0
    
    slope=np.zeros(np.shape(ai))
    ind=np.where(np.abs(refl)>0)[0]
    for iii in range(len(slope)-1):
        slope[iii+1]=ai[iii+1]-ai[iii]
    slope[ind]=0
    
    # for iii in range(0,len(ind)-1):
    #     if ind[iii+1]-ind[iii]>1:
    #         slope[ind[iii]+1:ind[iii+1]] = ai[ind[iii]+1]-ai[ind[iii]]
    return refl,slope


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap



#%% DISPLAYING DATA TOGETHER
cmap = plt.get_cmap('flag')
new_cmap = truncate_colormap(cmap, 0, 0.27)

dz1=z1[1]-z1[0]
dz3=z3[1]-z3[0]
print('dz1', dz1)
print('dz3', dz3)


shift=11

plt.figure(figsize=(15,8),dpi=80)
plt.imshow(seis[:,:].T, extent=[0,len(seis),zseis[-1],zseis[0]], 
           vmin=-1.2, vmax=1.2,cmap='gray',aspect='auto')


step, width = 2000, 850
ii=0
while ii*step<len(ai):
    deb=ii*step
    fin=ii*step+width
    if fin>len(ai):
        fin=len(ai)
    plt.imshow(ai[deb:fin,:].T, extent=[deb,fin,zai[-1]+shift,zai[0]+shift], 
                vmin=0,vmax=727714,cmap='turbo',aspect='auto')
    # Or use new_cmap instead of 'turbo'
    ii=ii+1

plt.ylim([80,30])
plt.xlim([0,6751])





