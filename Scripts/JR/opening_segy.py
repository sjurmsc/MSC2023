# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 10:40:31 2022

@author: JRD
"""


import segyio
import pandas as pd
import numpy as np

def opening_segy(filename):
    '''
    Parameters
    ----------
    filename : path to file
    
    Returns
    -------
    df : data frame with geometric information
    trace : 2D array with the data
    z : time vector associated with trace (in time if the segy file is in time)
    '''
    with segyio.open(filename, ignore_geometry=True) as segyfile:
        z=segyfile.samples
        segyfile.mmap()
        size=segyfile.trace.length
        
        # Get all header keys:
        header_keys = segyio.tracefield.keys
        # Initialize trace_headers with trace id as index and headers as columns
        df = pd.DataFrame(index=range(1, segyfile.tracecount+1), columns=header_keys.keys())
        
        # Fill dataframe with all trace headers values
        for k, v in header_keys.items():
            df[k] = segyfile.attributes(v)[:]
            
        # Keep only the necessary ones
        fields=['TRACE_SEQUENCE_LINE','CDP','SourceX','SourceY','GroupX','GroupY','CDP_X','CDP_Y','INLINE_3D','CROSSLINE_3D','ShotPoint']
        
        scaling=df['SourceGroupScalar'].unique()[0]
        if scaling<0:
            df['SourceX']=df['SourceX']/np.abs(scaling)
            df['SourceY']=df['SourceY']/np.abs(scaling)
            df['GroupX']=df['GroupX']/np.abs(scaling)
            df['GroupY']=df['GroupY']/np.abs(scaling)
            df['CDP_X']=df['CDP_X']/np.abs(scaling)
            df['CDP_Y']=df['CDP_Y']/np.abs(scaling)
        
        df=df[fields]
        df.rename(columns={ "TRACE_SEQUENCE_LINE":"group", "INLINE_3D": "inline", "CROSSLINE_3D": "xline"},inplace=True)
        
        # Removing columns with 0
        # df=df.loc[:, (trace_headers != 0).any(axis=0)]
        
        df['trace']='' 
        # trace=[]
        for i in np.arange(0,size):
            df.trace.at[i+df.index[0]]=segyfile.trace[i]
        #     trace.append(segyfile.trace[i])
        # trace=np.array(trace)
    return df,z


def load_segy_textual_header(filename):
    with segyio.open(filename, ignore_geometry=True) as segyfile:
        text = str(segyfile.text[0])
        # lines = map(''.join, zip( *[iter(text)] * 83))
        # for line in lines:
        #     print(line)
        
        text=text.split('              ')
        text=[txt for txt in text if len(txt)>0]
        
        ind_remove=[]
        for ii in range(len(text)):
            if text[ii].find('b\'C')>0:
                text[ii]=text[ii][text[ii].find('b\'C')+2:]
            elif text[ii].find('0 C')>0:
                text[ii]=text[ii][text[ii].find('0 C')+2:]
            else:
                ind_remove.append(ii)
        
        if len(ind_remove)>0:
            for ind in ind_remove[::-1]:
                del text[ind]
    return text