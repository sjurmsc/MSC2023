# Functions to be used for feature augmentation
import segyio
import json
from numpy import array

# Functions for loading data

def get_data_loc():
    d_filepath = './Data/data.json'
    return json.loads(d_filepath)

def get_traces(fp, mmap=True):
    """
    This function should conserve some information about the domain (time or depth) of
    the data.
    """
    with segyio.open(fp) as seis_data:
        z = seis_data.samples
        if mmap:
            seis_data.mmap()  # Only to be used if the file size is small compared to available memory
        traces = segyio.collect(seis_data.trace)
    return traces, z

def split_image_into_data_packets(traces, image_shape, mode='cut_lower', overlap=0):
    """
    Only Func i need before starting to train models

    overlap: The amount of traces that can overlap between the images
    """
    assert overlap < image_shape[0] # Allowing overlap of all but one trace
    if len(image_shape) == 1:
        lower_bound = traces.shape[1]
    else:
        lower_bound = image_shape[1]

    tracescount = traces.shape[0]
    delta = image_shape[0]-overlap

    X = []
    idx = [0, image_shape[0]]
    while idx[1] < tracescount:
        X.append(traces[idx[0]:idx[1]][:lower_bound])
        idx[0] += delta ; idx[1] += delta
    
    return array(X)




# Functions for augmenting the data