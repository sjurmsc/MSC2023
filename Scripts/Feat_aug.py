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
    with segyio.open(fp, ignore_geometry=True) as seis_data:
        z = seis_data.samples
        if mmap:
            seis_data.mmap()  # Only to be used if the file size is small compared to available memory
        traces = segyio.collect(seis_data.trace)
    return traces, z


def split_image_into_data_packets(traces, image_shape, dim=2, mode='cut_lower', upper_bound=0, overlap=0):
    """
    Only Func i need before starting to train models

    overlap: The amount of traces that can overlap between the images
    """
    width_shape, height_shape = image_shape
    assert overlap < width_shape  # Allowing overlap of all but one trace

    if len(image_shape) == 1:
        lower_bound = traces.shape[1]
    else:
        lower_bound = height_shape

    tracescount = traces.shape[0]
    delta = width_shape-overlap

    X = []
    idx = [0, width_shape]
    while idx[1] < tracescount:
        X.append(traces[idx[0]:idx[1], :])
        idx[0] += delta ; idx[1] += delta
    
    return array(X)


def pair_well_and_seismic():
    """
    For a given log trace, this function locates the appropriate seismic trace,
    and allows for adding the neighboring traces to output a seismic image
    centered at the well position. 
    """
    pass
