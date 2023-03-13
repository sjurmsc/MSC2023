# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 14:34:18 2021

@author: JRD
"""

import numpy as np
from scipy.signal import hilbert


def apply_attribute(X,attribute,**kwargs):
    if np.ndim(X)==2:
        Y=np.zeros(np.shape(X))
        if attribute=='envelop':
            for i in range(np.shape(X)[0]):
                Y[i,:]=envelop(X[i,:])
        elif attribute=='cumulative':
            for i in range(np.shape(X)[0]):
                Y[i,:]=cumulative(X[i,:])
        elif attribute=='energy':
            win=kwargs['win']
            for i in range(np.shape(X)[0]):
                Y[i,:]=energy(X[i,:],win)

    elif np.ndim(X)==3:
        Y=np.zeros(np.shape(X))
        for kk in range(np.shape(X)[-1]):
            if attribute=='envelop':
                for i in range(np.shape(X)[0]):
                    Y[i,:,kk]=envelop(X[i,:,kk])
            elif attribute=='cumulative':
                for i in range(np.shape(X)[0]):
                    Y[i,:,kk]=cumulative(X[i,:,kk])                       
            elif attribute=='energy':
                win=kwargs['win']
                for i in range(np.shape(X)[0]):
                    Y[i,:,kk]=energy(X[i,:,kk],win)
           
    
    if attribute=='similarity':
        samples=kwargs['samples']
        step_out=kwargs['step_out']
        lag=kwargs['lag']
        Y=similarity(X,samples,step_out, lag)
    return Y


def envelop(X):
    Y=hilbert(X)
    return np.abs(Y)


def energy(X,win):
    Y=np.zeros(len(X))
    for i in range(0+win//2,len(X)-win//2):    
        Y[i]=np.sqrt(np.sum(X[i-win//2:i+win//2]**2))/win
    return Y


def cumulative(X):
    return np.cumsum(X)


def difference(X,Y):
    diff=X/X.max()-Y/Y.max()
    return diff


def similarity(traces, samples=20, step_out=1, lag=0):
    """
    Compute similarity for each point of a seismic section using a variance method between traces.
    For each point, a kernel of n_samples length is extracted from a trace. 
    The similarity is calculated as a normalized variance between two adjacent trace sections, 
    where a value of 1 is obtained by identical if the traces are identical. 
    The step out will decide how many adjacent traces will be used for each kernel,
    and should be increased for poor quality data. 
    The lag determines how much neighbouring traces can be shifted when calculating similiarity, 
    which should be increased for dipping data.
    
    :param traces: A 2D numpy array arranged as [time, trace].
    :param samples: the length in samples of the window trace kernel used to calculate the similarity.
    :keyword step_out (default=1 ):
        The number of adjacent traces to the kernel to check similarity. 
        The maximum similarity value will be chosen.
    :keyword lag (default=0):
        The maximum number of time samples adjacent traces can be shifted by. The maximum similarity of will be used.
    """
    half_n_samples = int(samples/2)

    similarity_cube = np.zeros_like(traces, dtype='float32')
    traces = np.nan_to_num(traces)

    for j in np.arange(-lag, lag+1):

        for i in (np.arange(step_out)):
            for idx in range(similarity_cube.shape[0]):

                # Get the signal
                start_sig_idx = max(0, (idx+(j*(i+1))-half_n_samples))
                stop_sig_idx = min(similarity_cube.shape[0]-1, (idx-((i+1)*j))+half_n_samples)

                # Get the data
                start_data_idx = max(0, (idx - half_n_samples))
                end_data_idx = start_data_idx + (stop_sig_idx - start_sig_idx)

                if(end_data_idx > traces.shape[0]):
                    break

                signal = traces[start_sig_idx:stop_sig_idx, :]
                data = traces[start_data_idx:end_data_idx, :]

                squares = (signal*signal).sum(axis=0)

                squares_of_diff = ((signal[:,1+i:] - data[:, :-(1+i)])**2.).sum(axis=0)

                squares[squares == 0.0] = 0.001
                squares_of_diff[squares_of_diff == 0.0] = 0.001
                sim = 1.0 - np.sqrt(squares_of_diff) / ((np.sqrt(squares[1+i:]) + np.sqrt(squares[:-(1+i)]) ))

                similarity_cube[idx, (i+1):] = np.maximum(sim,
                                                         similarity_cube[idx, (i+1):])
    return similarity_cube





