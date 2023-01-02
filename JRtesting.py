# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 11:00:48 2023

@author: JRD
"""

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input, Model

from ArchitecturesJRD import *

import matplotlib.pyplot as plt
import numpy as np
import time

from keras.utils.vis_utils import plot_model




ss=20*50
(Xtrain, ytrain), (Xtest, ytest) = mnist.load_data()
ytrain=to_categorical(ytrain)
ytest=to_categorical(ytest)

Xtrain=np.expand_dims(Xtrain[:ss],axis=-1)
Xtest =np.expand_dims(Xtest[:ss] ,axis=-1)
ytrain=ytrain[:ss]
ytest =ytest[:ss]


Xtrain=Xtrain[:,:9,:9,:]
tic=time.time()

pad='valid'
ps=2
    
# Creating the ML network
input_layer=Input(shape=Xtrain.shape[1:])   
x = TCN(nb_filters=8,
        kernel_size=(2,2),
        nb_stacks=1,
        dilations=(1,2,4),
        padding='same',
        use_skip_connections=True,
        dropout_type = 'spatial',
        dropout_rate=0,
        return_sequences=True,
        activation='relu',
        convolution_func= Conv2D,
        )(input_layer)


reg = CNN(nb_filters=4,
        kernel_size=2,
        nb_stacks=2,
        name = 'Regression_module'
        )(x)   

reg = Conv2D(1, kernel_size=2, padding='same', activation='linear', name='regression_output')(reg)


model = Model(inputs=input_layer, outputs=reg)
plot_model(model,show_shapes=True)


tmp=model.get_layer(index=1)
plot_model(tmp,show_shapes=True)
# tmp.get_config()


tmp=model.get_layer(index=1).get_layer(index=0)
tmp.Conv2D_0.filters
tmp.Conv2D_0.get_config()
