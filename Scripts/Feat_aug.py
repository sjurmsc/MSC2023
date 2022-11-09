"""
Functions to be used for feature augmentation
"""
import segyio
import json
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from os import listdir
from pathlib import Path

# Functions for loading data

def get_data_loc():
    d_filepath = './Data/data.json'
    return json.loads(d_filepath)


def get_traces(fp, mmap=True, zrange: tuple = (None,), length: int = None, ztrunc=100):
    """
    This function should conserve some information about the domain (time or depth) of
    the data.
    """
    with segyio.open(fp, ignore_geometry=True) as seis_data:
        z = seis_data.samples
        if mmap:
            seis_data.mmap()  # Only to be used if the file size is small compared to available memory
        traces = segyio.collect(seis_data.trace)
    # if zrange[0] != None:
    #     pass
        #zmin = z[]
    traces, z = traces[:, z<ztrunc], z[z<ztrunc]
    return traces, z


def split_image_into_data_packets(traces, width_shape=7, dim=2, mode='cut_lower', upper_bound=0, overlap=0):
    """
    Only Func i need before starting to train models

    overlap: The amount of traces that can overlap between the images
    """
    
    assert overlap < width_shape, 'Overlap cannot excede the with of the seismic image'
    
    tracescount = traces.shape[0]
    delta = width_shape-overlap

    X = []
    idx = [0, width_shape]
    while idx[1] < tracescount:
        X.append(traces[idx[0]:idx[1], :])
        idx[0] += delta ; idx[1] += delta
    
    return array(X)


def sgy_to_keras_dataset(X_data_label_list,
                         y_data_label_list,
                         test_size, 
                         feature_dimension = 1,
                         zrange: tuple = (None,), 
                         validation = False, 
                         normalize = 'MinMaxScaler',
                         random_state=1):
    """
    random_state may be passed for recreating results
    """
    data_dict = load_data_dict()
    dataset = [] # Features within must be in the shape X[,y]


    # Something to evaluate that z is same for all in a feature
    
    for key in X_data_label_list:
        data_dir = Path(data_dict[key])
        for fname in data_dir.glob('*.sgy'):
            trace, z = get_traces((data_dir / fname), zrange=zrange)
        # Add the data to the dataset, must be robust for AI, CPT or seismic data
    
    dataset = array(dataset)

    
    
    # Normalization
    if normalize == 'MinMaxScaler':
        scaler = MinMaxScaler()
        X_new = scaler.fit_transform(X, y)
    
    train_dataset, test_dataset = train_test_split(dataset, test_size=test_size, random_state=random_state)  # dataset must be np.array
    
    if validation:
        test_dataset, val = train_test_split(test_dataset, test_size=test_size, random_state=random_state)
        return train_dataset, test_dataset, val
    return train_dataset, test_dataset


def collect_sgy_data_in_dataset():
    pass

def match_files(X_folder_loc, y_folder_loc, file_extension='.sgy'):
    """
    Matches the features from two seperate traces by filename
    Filenames corresponding to each other are returned as a list
    of tuples in the form [(X_file, y_file)]
    """
    X_dir = Path(X_folder_loc); y_dir = Path(y_folder_loc)

    X_prefixes = []
    for fname in X_dir.glob('*'+file_extension):
        [fname[fname.find('_', 3)]]


def pair_well_and_seismic():
    """
    For a given log trace, this function locates the appropriate seismic trace,
    and allows for adding the neighboring traces to output a seismic image
    centered at the well position. 
    """
    pass


def load_data_dict():
    data_json = './Data/data.json'
    with open(data_json, 'r') as readfile:
        data_dict = json.loads(readfile.read())
    return data_dict


def update_data_dict():
    """ Edit this funciton to change the filepaths to the relevant data
        Filepaths must be relative to the repository, which is in user folder.
        Double dot (..) steps outside of this folder to access the OneDrive
        folder
    """
    data_json = './Data/data.json'
    root = '../OneDrive - NGI/Documents/NTNU/MSC_DATA/'
    data_dict = {
        '00_AI'                 : root + '00_AI',
        '2DUHRS_06_MIG_DEPTH'   : root + '2DUHRS_06_MIG_DEPTH'
    }
    with open(data_json, 'w') as writefile:
        writefile.write(json.dumps(data_dict, indent=2))


def combine_seismic_traces():
    pass


def find_nth(haystack, needle, n):
    n = abs(n)
    max_needle_amount = len(haystack.split(needle))-1; assert n < max_needle_amount
    if n:
        loc = find_nth(haystack[haystack.find(needle)+1:], needle, n-1)
    else:
        return len(haystack[haystack.find(needle):])
    return len(haystack) - loc

def format_input_output(dataset):
    regression_data, seismic_data = dataset
    X = seismic_data; Y = dataset
    