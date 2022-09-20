# Functions to be used for feature augmentation
import segyio
import json

# Functions for loading data

def get_data_loc():
    d_filepath = './Data/data.json'
    return json.loads(d_filepath)

def get_traces(fp):
    with segyio.open(fp) as seis_data:
        traces = segyio.collect(seis_data.trace)
    return traces

def split_image_into_data_packets(traces, image_shape=False):
    """
    Only Func i need before starting to train models
    """
    if image_shape:
        #traces
        pass




# Functions for augmenting the data