"""
Initializes the task to be done on Odin

runs all code that has to do with different data permutations
"""
import json

settings = {}

dropout=param['dropout']
    ks=param['kernel_size']
    filters=param['filters']
    loss=param['loss']
    dilation=param['dilation']
    lr=param['learn_rate']

control = {}
control['settings'] = {}
control['summary_stats'] = {}
control['data'] = []

print(control)