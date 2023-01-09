# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 15:26:21 2022

@author: JRD
"""

from opening_segy import *
import os
import matplotlib.pyplot as plt

tf.keras.backend.clear_session()

# LOADING FILENAMES
# path_seis=os.path.join('P:','2019','07','20190798','Background','2DUHRS_06_MIG_DEPTH')
path=os.path.join('C:\\','Users','JRD','OneDrive - NGI','Python project','TCN-Marmousi','TNW data')

fileai=['TNWX_BX06_5600_MIG.Abs_Zp.sgy']
fileseis=['TNWX_BX06_5600_MIG.sgy']
# fileseis, fileai = [],[]
# for ff in os.listdir(path):
#     if ff[-4:]=='.sgy':
#         if 'Zp' in ff:
#             fileai.append(ff)
#         else:
#             fileseis.append(ff)


# LOADING DATA
dfseis,zseis=opening_segy(os.path.join(path,fileseis[0]))
dfai,zai=opening_segy(os.path.join(path,fileai[0]))


seis=np.stack(dfseis.trace.to_numpy())
ai=np.stack(dfai.trace.to_numpy())

aitmp=ai.copy()

#%% CORRECTION OF DATA AND DISPLAY
ai=aitmp.copy()

# CORRECT NEGATIVE AI
# IF AT BEGINNING - SET TO 0
ind=np.where(np.where(ai<0)[1]<120)[0]
ai[np.where(ai<0)[0][ind],np.where(ai<0)[1][ind]]=0
# IF AT THE END SETUP THE LAST "VALID" AI FOR THE REST OF THE TRACE
for i in range(len(ai)):
    ind=np.where( ai[i,500:]<10 )[0][0]+500
    ai[i,ind-5:]=ai[i,ind-20:ind-3].mean()


# shift=0
shift=13
# DISPLAY SEISMIC TRACE AND ACOUSTIC IMPEDANCE
plt.figure(figsize=(15,8),dpi=80)
plt.imshow(seis[:,:].T, extent=[0,len(seis),zseis[-1],zseis[0]], vmin=-1.2, vmax=1.2,cmap='gray',aspect='auto')
plt.imshow(ai[:,:].T, extent=[0,len(ai),zai[-1]+shift,zai[0]+shift], cmap='turbo',aspect='auto',alpha=0.4)
# plt.ylim(zai[-1]+shift,zai[0]+shift)
plt.ylim(105,20)
# plt.ylim(95,35)
# plt.xlim(10000,15000)




# plt.figure(figsize=(15,8),dpi=120)
# # plt.imshow(ai[590:1000,:].T,cmap='turbo',aspect='auto')
# plt.imshow(ai[:,:].T,extent=[0,len(ai),zai[-1]+shift,zai[0]+shift], cmap='turbo',aspect='auto')
# # plt.ylim(95,70)
# plt.ylim(105,38)
# # plt.xlim(10000,15000)





#%% Preparing data for ML
zaish=zai+shift

depthstart,depthend=42,115
aiML=ai[:,np.where(zaish==depthstart)[0][0]:np.where(zaish==depthend)[0][0]]
seisML=seis[:,np.where(zseis==depthstart)[0][0]:np.where(zseis==depthend)[0][0]]

# DISPLAY SEISMIC TRACE AND ACOUSTIC IMPEDANCE
# plt.figure(figsize=(15,8),dpi=120)
# plt.imshow(seisML[:,:].T,extent=[0,len(aiML),depthend,depthstart], vmin=-0.8, vmax=0.8,cmap='gray',aspect='auto')

# plt.figure(figsize=(15,8),dpi=120)
# plt.imshow(aiML[:,:].T, extent=[0,len(aiML),depthend,depthstart],cmap='turbo',aspect='auto')

nb_traces=len(aiML)
trainratio=1/100
size_training=int(nb_traces*trainratio)
size_test=4*size_training

indextrain=np.random.randint(0,nb_traces-1,size_training)
indextest=np.random.randint(0,nb_traces-1,size_test)


# borne is to delete the beginning of the profiles, that are constant value
aitrain=aiML[indextrain,:]
seistrain=seisML[indextrain,:]
aitest=aiML[indextest,:]
seistest=seisML[indextest,:]


# Scaling the acoustic impedance
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# Define the type of scaling
scalerai=MinMaxScaler()  

# Fitting the scaler on the training set
scalerai.fit(aitrain.reshape(-1,1))

# using the scaler to transform the trainingset and the testset
aitrain_scal=scalerai.transform(aitrain.reshape(-1,1)).reshape(aitrain.shape)
aitest_scal=scalerai.transform(aitest.reshape(-1,1)).reshape(aitest.shape)


# Scaling the seismic
scalerseis=StandardScaler()  
scalerseis.fit(seistrain.reshape(-1,1))

# seistrain_scal=scalerseis.transform(seistrain.reshape(-1,1)).reshape(seistrain.shape)
# seistest_scal=scalerseis.transform(seistest.reshape(-1,1)).reshape(seistest.shape)

# seistrain_scal=seistrain
# seistest_scal=seistest

# Expanding the dimensions to fit the requirement of the ML [size training, nb samples, nb features]
X = np.expand_dims(seistrain,axis=np.ndim(seistrain))
y = np.expand_dims(aitrain_scal,axis=np.ndim(aitrain_scal))
testX = np.expand_dims(seistest,axis=np.ndim(seistest))
testy = np.expand_dims(aitest_scal, axis=np.ndim(aitest_scal))



#%%
# ML Structure
param={}
param['dropout']=0.05    # dropout to reduce risk of overfitting (percent)
param['kernel_size']=5   # size of the kernel for the convolution operations - usually between 3 and 5
param['filters']=[3,5,5,5,6,6,6,6]  # nb filters per convolution layers - need to have the same number of value as the dilation factor
param['loss']='mse'      # loss function, could be mse, mae, ...
param['dilation']=[1,2,4,8,16,32,64,128]  # dilation factor - "subsampling" of the data between each convolution layers
param['learn_rate']=0.001
param['skip']=False     # skip connection
param['noise_on_trainset']=0.1


from TCN_architecture import *
model = TCNrelu(trainX=X, param=param)



#%% Training the ML
import time
E=300

tic=time.time()
history=model.fit(X,y,validation_data=(testX,testy),batch_size=32,epochs=E,verbose=1)
print('time:  ', str( round(time.time()-tic)), ' sec')

# Display the metrics through the training process
plt.plot(history.history['mse']) # on the training set
plt.plot(history.history['val_mse']) # on the validation set
plt.yscale('log')



#%%

# input_prediction = scalerseis.transform(seisML.reshape(-1,1)).reshape(seisML.shape)
input_prediction = seisML
input_prediction = np.expand_dims(input_prediction,axis=np.ndim(input_prediction))


# Predicting the acoustic impedance
ai_prediction = model.predict(input_prediction)[:,:,0]

# Scaling back the prediction
ai_prediction=scalerai.inverse_transform(ai_prediction.reshape(-1,1)).reshape(ai_prediction.shape)

# displaying the true acoustic impedance and the predicted one
f,ax=plt.subplots(2,1,figsize=(15,15),dpi=120)
ax[0].imshow(aiML.T,cmap='turbo',aspect='auto',vmin=0,vmax=705000)
ax[1].imshow(ai_prediction.T,cmap='turbo',aspect='auto',vmin=0,vmax=705000)


ss=0
resid=aiML-ai_prediction[ss:len(aiML)+ss]
print('mse ', np.sqrt(np.mean(resid**2)))

plt.figure(figsize=(15,8),dpi=120)
plt.imshow(resid.T,cmap='turbo', aspect='auto')




#%%

ind=[100,5000,10000,15000,2000,25000]

plt.figure(figsize=(15,20),dpi=130)
maxx=np.max(aiML)
i=0
for ii in ind:
    plt.plot(aiML[ii]/maxx+i,color='C0')
    plt.plot(ai_prediction[ii]/maxx+i,color='C1')
    i+=0.5


#%%
runcell(0, 'C:/Users/JRD/OneDrive - NGI/Python project/TCN-Marmousi/TNW data/open_display.py')
runcell(1, 'C:/Users/JRD/OneDrive - NGI/Python project/TCN-Marmousi/TNW data/open_display.py')
runcell(2, 'C:/Users/JRD/OneDrive - NGI/Python project/TCN-Marmousi/TNW data/open_display.py')
runcell(3, 'C:/Users/JRD/OneDrive - NGI/Python project/TCN-Marmousi/TNW data/open_display.py')
runcell(4, 'C:/Users/JRD/OneDrive - NGI/Python project/TCN-Marmousi/TNW data/open_display.py')
runcell(5, 'C:/Users/JRD/OneDrive - NGI/Python project/TCN-Marmousi/TNW data/open_display.py')
runcell(6, 'C:/Users/JRD/OneDrive - NGI/Python project/TCN-Marmousi/TNW data/open_display.py')



