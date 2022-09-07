"""
Initializes the task to be done on Odin

runs all code that has to do with different data permutations
"""
import json
import time


settings = {}

dropout=param['dropout']
ks=param['kernel_size']
filters=param['filters']
loss=param['loss']
dilation=param['dilation']
lr=param['learn_rate']

control = {}
control['settings'] = {} # settings
control['summary_stats'] = {} # to be filled in later
control['data'] = [] # where to retrieve the dataset from

print(control)