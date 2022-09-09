"""
Initializes the task to be done on Odin

this script will contain all instructions for which networks to run on odin
--runs all code that has to do with different data permutations

All git operations will happen in this script
"""
# External packages
import json
import time
from pathlib import Path

# My scripts
from Log import *
from Architectures import *

class RunModel:
    """
    Takes settings and runs a model based on it
    """
    def __init__(settings):
        pass


settings = {}
"""
In settings must be:
network: f.ex 2DTCN, 1DTCN, 2DTCN_WS [weight sharing], Randomforest ... etc
# What the network is, will affect what the init script uses to initialize the model

if TCN networks:
    dropout:
    kernel_size
    filters
    loss
    dilation
    

for all networks:
    dataset: contains the label names that can be used as keys for retrieving file paths
            for getting the data [file paths must be robust, and located either on p: or in the
            repo]
    epochs: 
    batches:
    learn_rate:

for training where certain fields are not needed, these may be filled with None, or be omitted
"""


control = {}
control['settings'] = settings.copy() # settings
control['summary_stats'] = {} # to be filled in later

if __name__ == '__main__':
    # Her skal koden gå
    ai = r'P:\2019\07\20190798\Deliverables\Digital_Deliverables\00_AI\TNW_B04_3360_MIG.Abs_Zp.sgy'
    data = {}
    data['TNW_AI'] = ai
    f = json.dumps(data)
    with open('Data\\data.json', 'w') as wfile:
        wfile.write(f)
