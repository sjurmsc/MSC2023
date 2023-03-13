"""
Functions to be used for feature augmentation
"""
import segyio
import json
from numpy import array, row_stack, intersect1d, where, amax, amin
from pandas import read_csv
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from os import listdir
from pathlib import Path
import sys
# from JR.Seismic_interp_ToolBox import ai_to_reflectivity, reflectivity_to_ai

# Functions for loading data

def get_data_loc():
    d_filepath = './Data/data.json'
    return json.loads(d_filepath)


def get_cpt_data_from_file(fp, zrange: tuple = (None, 100)):
    """
    Function to read in CPT data from a file
    """
    cols = ['Depth', 'cone', 'friction', 'pore 2']

    # Read in data and replace empty values with NaN
    data = read_csv(fp, sep=';')[cols].iloc[1:-1]
    # data.applymap(lambda x: x.replace('', 'nan'))
    
    # Replace empty strings in data with NaN
    # data = data.applymap(lambda x: np.nan if x == '' else x)

    data = array(data.replace(r'^\s*$', np.nan, regex=True)).astype(float)
    
    z = data[:, 0]
    data = data[:, 1:]

    
    data = data[(z>=zrange[0])&(z < zrange[1])]
    z = z[(z>=zrange[0])&(z < zrange[1])]

    return data, z

from pandas import read_excel

def match_cpt_to_seismic(n_neighboring_traces=0, zrange: tuple = (30, 100)):
    distances = r'C:\Users\SjB\OneDrive - NGI\Documents\NTNU\MSC_DATA\Distances_to_2Dlines.xlsx'
    CPT_match = read_excel(distances)


    cpt_dict = get_cpt_las_files()

    match_dict = {}


    for i, row in CPT_match.iterrows():
        # For testing
        if i > 10: break

        cpt_loc = int(row['Location no.'])
        print('Retrieving from TNW{:03d}'.format(cpt_loc))
        CDP = int(row['CDP'])
        seis_file = '../OneDrive - NGI/Documents/NTNU/MSC_DATA/2DUHRS_06_MIG_DEPTH/{}.sgy'.format(row['2D UHR line'])
        with segyio.open(seis_file, ignore_geometry=True) as SEISMIC:
            z = SEISMIC.samples
            a = array(SEISMIC.attributes(segyio.TraceField.CDP)).astype(int)
            traces = segyio.collect(SEISMIC.trace)[where(abs(a - CDP) <= n_neighboring_traces)[0]]
            traces, z_traces = traces[:, (z>=zrange[0])&(z<zrange[1])], z[(z>=zrange[0])&(z<zrange[1])]
        cpt_name = f'TNW{cpt_loc:03d}'
        CPT_DATA = cpt_dict[cpt_name].values
        CPT_DEPTH = cpt_dict[cpt_name].index.values

        # TEMPORARY
        SEAFLOOR = 35

        match_dict[cpt_loc] = {'CDP': CDP, 
                               'CPT_data': CPT_DATA, 
                               'Seismic_data': traces, 
                               'z_traces': z_traces, 
                               'z_cpt': CPT_DEPTH,
                               'seafloor': SEAFLOOR}
    return match_dict
        

def create_sequence_dataset(n_neighboring_traces=5, zrange: tuple = (30, 100), random_flip=True, shortest_sequence=5, ran=406):
    """
    Creates a dataset with sections of seismic image and corresponding CPT data where
    none of the CPT data is missing.
    """
    match_dict = match_cpt_to_seismic(n_neighboring_traces, zrange)
    X, y = [], []
    for key, value in match_dict.items():
        z_GM = np.arange(0, value['z_cpt'].max(), 0.1)
        cpt_vals = value['CPT_data']
        bootstraps, z_GM = bootstrap_CPT_by_seis_depth(cpt_vals, value['z_cpt'], z_GM, n=20)
        sea_floor_depth = value['seafloor']
        correlated_cpt_z = z_GM + sea_floor_depth

        for bootstrap in bootstraps:
            # Split CPT data by nan values
            row_w_nan = lambda x: np.any(np.isnan(x))
            nan_idx = np.apply_along_axis(row_w_nan, axis=1, arr=bootstrap)
            split_indices = np.unique([(i+1)*b for i, b in enumerate(nan_idx)])[1:]
            splits = np.split(bootstrap, split_indices)
            splits_depth = np.split(correlated_cpt_z, split_indices)

            for section, z in zip(splits, splits_depth):

                if (section.shape[0]-1) > shortest_sequence:
                    in_seis = np.where((value['z_traces'] >= z.min()) & (value['z_traces'] < z.max()))
                    seis_seq = value['Seismic_data'][:, in_seis][:, 0, :]
                    cpt_seq = section[:-1, :]
                    assert seis_seq.shape[1] == 2*cpt_seq.shape[0], 'Seismic sequences must represent the same interval as the CPT'
                    X.append(seis_seq) # .reshape((seis_seq.shape[0], seis_seq.shape[1], 1)))
                    y.append(cpt_seq) # .reshape((1, cpt_seq.shape[0], cpt_seq.shape[1])))
            

    # Randomly flip the X data about the 1 axis
    if random_flip:
        for i in range(len(X)):
            if np.random.randint(2):
                X[i] = np.flip(X[i], 0)
    
    

    return X, y




def get_traces(fp, mmap=True, zrange: tuple = (None, 100)):
    """
    This function should conserve some information about the domain (time or depth) of
    the data.
    """
    with segyio.open(fp, ignore_geometry=True) as seis_data:
        z = seis_data.samples
        if mmap:
            seis_data.mmap()  # Only to be used if the file size is small compared to available memory
        traces = segyio.collect(seis_data.trace)
    
    traces, z = traces[:, z<zrange[1]], z[z<zrange[1]]
    return traces, z


def get_matching_traces(fp_X, fp_y, mmap = True, zrange: tuple = (25, 100), group_traces: int = 1, trunc = False):
    """
    Function to assist in maintaining cardinality of the dataset
    %%%%%%%%%%%%%%%% Can add overlap here
    """
    assert group_traces%2, 'Amount of traces must be odd to have a center trace'

    with segyio.open(fp_X, ignore_geometry=True) as X_data:
        with segyio.open(fp_y, ignore_geometry=True) as y_data:
            # retrieving depth values for target and input data
            z_X = X_data.samples
            z_y = y_data.samples

            # getting index of max depth for truncation
            X_max_idx = amax(where(z_X < zrange[1])) + 1
            y_max_idx = amax(where(z_y < zrange[1])) + 1

            # The acoustic impedance starts at depth 25m
            X_min_idx = amin(where(z_X >= z_y[0]))


            if mmap: X_data.mmap(); y_data.mmap() # initiate mmap mode for large datasets

            # get information about what traces are overlapping
            nums_X = segyio.collect(X_data.attributes(segyio.TraceField.CDP))
            nums_y = segyio.collect(y_data.attributes(segyio.TraceField.CDP))
            CDP, idx_X, idx_y = intersect1d(nums_X, nums_y, return_indices=True)
            assert len(idx_X) == len(idx_y)

            # collect the data
            X_traces = segyio.collect(X_data.trace)[idx_X, X_min_idx:X_max_idx]
            y_traces = segyio.collect(y_data.trace)[idx_y, :y_max_idx]

            X_func = interp1d(z_X[X_min_idx:X_max_idx], X_traces, kind='cubic', axis=1)
            X_traces = X_func(z_y[:y_max_idx])
 
            # y_refl, slope = ai_to_reflectivity(y_traces)
            # y_interp = interp1d(z_y[:y_max_idx], y_refl, kind='nearest', axis=1)
            # y_interp_refl = array(y_interp(z_X[X_min_idx:X_max_idx]))
            # y_traces = reflectivity_to_ai(y_interp_refl, slope)

    if not group_traces == 1:
        num_traces = X_traces.shape[0]
        len_traces = X_traces.shape[1]
        num_images = num_traces//group_traces
        indices_truncated = num_images*group_traces
        discarded_images = num_traces-indices_truncated
        l_indices = (discarded_images//2); r_indices = indices_truncated + l_indices + discarded_images%2
        X_traces = X_traces[l_indices:r_indices].reshape((num_images, group_traces, len_traces))
        y_traces = y_traces[l_indices:r_indices].reshape((num_images, group_traces, len_traces))
    
    if trunc:  # Done as a quick way to remove bad data, as it is most often at the ends
        X_traces = X_traces[trunc:-trunc]
        y_traces = y_traces[trunc:-trunc]
    
    return X_traces, y_traces, (z_X, z_y)


def sgy_to_keras_dataset(X_data_label_list,
                         y_data_label_list,
                         test_size=0.2, 
                         group_traces = 1,
                         zrange: tuple = (None, 100), 
                         reconstruction = True,
                         validation = False, 
                         X_normalize = None,
                         y_normalize = 'MinMaxScaler',
                         random_state=1,
                         shuffle=True,
                         min_y = 0.,
                         fraction_data=False,
                         truncate_data=False):
    """
    random_state may be passed for recreating results
    """
    data_dict = load_data_dict()

    # Something to evaluate that z is same for all in a feature
    X = array([]); y = array([])

    for i, key in enumerate(X_data_label_list):
        x_dir = Path(data_dict[key])
        y_dir = Path(data_dict[y_data_label_list[i]])
        matched = match_files(x_dir, y_dir)
        if fraction_data: matched = matched[:int(len(matched)*fraction_data)] # %%%%%%%%%%%%%%%%%%%%%quickfix
        m_len = len(matched)
        for i, (xm, ym) in enumerate(matched):
            # Giving feedback to how the collection is going
            sys.stdout.write('\rCollecting trace data into dataset {}/{}'.format(i+1, m_len))
            sys.stdout.flush()

            try: x_traces, y_traces, z = get_matching_traces(xm, ym, zrange=zrange, group_traces=group_traces, trunc=truncate_data)
            except: print('\nCould not load file {}\n'.format(xm)); continue

            if not len(X):
                X = array(x_traces)
                y = array(y_traces)
            else:
                X = row_stack((X, x_traces))
                y = row_stack((y, y_traces))
    sys.stdout.write('\n'); sys.stdout.flush()
    
    y[where(y<min_y)] = min_y

    X_scaler = None
    y_scaler = None
    # Normalization
    if X_normalize == 'MinMaxScaler':
        X_scaler = MinMaxScaler()
        X_new = X_scaler.fit_transform(X.reshape(-1, 1))
        X = X_new.reshape(X.shape)
    elif X_normalize == 'StandardScaler':
        X_scaler = StandardScaler()
        X_new = X_scaler.fit_transform(X.reshape(-1, 1))
        X = X_new.reshape(X.shape)
    if y_normalize == 'MinMaxScaler':
        y_scaler = MinMaxScaler()
        y_new = y_scaler.fit_transform(y.reshape(-1, 1))
        y = y_new.reshape(y.shape)


    train_X, test_X, train_y, test_y = train_test_split(X, y, 
                                                        test_size=test_size, 
                                                        random_state=random_state,
                                                        shuffle=shuffle)  # dataset must be np.array
    
    if validation:
        test_X, val_X, test_y, val_y = train_test_split(test_X, test_y, test_size=test_size, random_state=random_state, shuffle=shuffle)
        if reconstruction:
            train_y = [train_y, train_X]
            test_y = [test_y, test_X]
            val_y = [val_y, val_X]
        return (train_X, train_y), (test_X, test_y), (val_X, val_y)
    
    if reconstruction:
        train_y = [train_y, train_X]
        test_y = [test_y, test_X]
    return (train_X, train_y), (test_X, test_y), (X_scaler, y_scaler)


def match_files(X_folder_loc, y_folder_loc, file_extension='.sgy'):
    """
    Matches the features from two seperate traces by filename
    Filenames corresponding to each other are returned as a list
    of tuples in the form [(X_file, y_file)]
    """
    X_dir = list(Path(X_folder_loc).glob('*'+file_extension)); y_dir = list(Path(y_folder_loc).glob('*'+file_extension))

    file_pairs = []
    for i, fname in enumerate(X_dir):
        X_name = fname.name[:fname.name.find('_MIG_DPT.sgy')]
        j_list = []
        for j, yfile in enumerate(y_dir):
            y_name = yfile.name[:yfile.name.find('_MIG.Abs_Zp.sgy')]
            if X_name == y_name:
                file_pairs.append((str(fname), str(yfile)))
                j_list.append(j)
        [y_dir.pop(j) for j in j_list]
    return file_pairs
          

def pair_well_and_seismic():
    """
    For a given log trace, this function locates the appropriate seismic trace,
    and allows for adding the neighboring traces to output a seismic image
    centered at the well position. 
    """
    pass


class CPT_TRACE:

    def __init__(self, name, coords, trace, z):
        self.name = name
        self.lon = coords[0]
        self.lat = coords[1]
        self.trace = trace
        self.z = z


import numpy as np
import pandas as pd

def bootstrap_CPT_by_seis_depth(cpt_data, cpt_depth, GM_depth, n=1000, plot=False, to_file=''):
    """ This function creates bins of cpt values at ground model depths, and then samples
        from these bins to create a new downsampled CPT dataset. This is done to
        increase the number of CPT samples at model depths. The function returns
        a new CPT dataset with the same number of samples as the ground model dataset."""
    cpt_data = np.array(cpt_data)
    cpt_depth = np.array(cpt_depth)
    GM_depth = np.array(GM_depth)

    # Making sure that GM_depth is evenly spaced, and setting half bin size
    assert np.all(np.isclose(np.sum(np.diff(np.diff(GM_depth))), 0))
    half_bin = np.diff(GM_depth)[0]/2

    assert cpt_data.shape[0]==cpt_depth.shape[0]

    # Create bins of CPT values at GM depths
    cpt_bins = []
    for i, d in enumerate(GM_depth):
        cpt_bins.append(cpt_data[np.where((cpt_depth>=d-half_bin)&(cpt_depth<d+half_bin))])

    # Sample from bins
    cpt_samples = np.array([]).reshape((n, 0, 3))
    for i, b in enumerate(cpt_bins):
        if not b.size: cpt_samples = np.append(cpt_samples, np.full((n, 1, 3), np.nan), axis=1)
        else: 
            # pick n random rows of samples from b and append to cpt_samples
            cpt_samples = np.append(cpt_samples, b[np.random.choice(b.shape[0], n, replace=True), :].reshape((n, 1, 3)), axis=1)

    # cpt_samples = np.array(cpt_samples)
    # cpt_samples = cpt_samples.reshape((len(cpt_samples)*n, 1))

    if plot:

        # plot cpt data in three subplots
        fig, ax = plt.subplots(1, 3, sharey=True)
        for trace in cpt_samples:
            ax[0].plot(trace[:, 0], GM_depth, '--.')
            ax[1].plot(trace[:, 1], GM_depth, '--.')
            ax[2].plot(trace[:, 2], GM_depth, '--.')

        ax[0].set_title('$q_c$')
        ax[1].set_title('$f_s$')
        ax[2].set_title('$u_2$')

        # Title the figure
        plt.suptitle('Bootstrapped CPT')

        # invert y axis
        for a in ax:
            a.invert_yaxis()

        # plt.hist(cpt_samples, bins=100)'
        if len(to_file):
            plt.savefig(to_file)
        else:
            plt.show()
    plt.close()
    return cpt_samples, GM_depth
    
import lasio
def get_cpt_las_files(cpt_folder_loc='../OneDrive - NGI/Documents/NTNU/MSC_DATA/combined'):
    """
    Returns a list of all las files in a given folder
    """
    cpt_dir = list(Path(cpt_folder_loc).glob('*.las'))
    cpt_keys = [cpt.name[:cpt.name.find('.las')] for cpt in cpt_dir]

    df_dict = {}

    for key, cpt in zip(cpt_keys, cpt_dir):
        with open(cpt, 'r') as lasfile:
            las = lasio.read(lasfile)
            df = las.df().iloc[:, :3]
            df_dict[key] = df.copy()
            # bootstrap_CPT_by_seis_depth(df.values, df.index.values, np.arange(0, df.index.values.argmax(), 0.1), n=10, plot=True, to_file=f'./Assignment figures/Bootstrap_figs/{key}.png')

    return df_dict


def get_seismic_where_there_is_cpt(cpt, z_cpt, seis, z_seis):
    """
    For a given cpt trace, this function gives a pair of seismic image and cpt for a continuous range where all cpt parameters are defined.
    """
    # Get the indices of the cpt trace where all parameters are defined
    idx = np.where(~np.isnan(cpt).any(axis=1))[0]
    # Get the corresponding cpt depth
    z_cpt = z_cpt[idx]
    # Get the corresponding cpt trace
    cpt = cpt[idx]
    # Get the corresponding seismic trace
    seis = seis[idx]
    # Get the corresponding seismic depth
    z_seis = z_seis[idx]

    return cpt, z_cpt, seis, z_seis


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


def find_nth(haystack, needle, n : int):
    n = abs(n); assert n > 0
    max_needle_amount = len(haystack.split(needle)); assert n < max_needle_amount
    if n-1:
        intermed = haystack.find(needle) + 1
        loc = intermed + find_nth(haystack[intermed:], needle, n-1)
    else:
        return haystack.find(needle)
    return loc


def find_duplicates(m_files):
    dupes = dict()
    names = []
    for i, (X_m, y_m) in enumerate(m_files):

        name = X_m[62:-12]
        name_list = name.split('_')
        for n in names:
            n_list = n.split('_')
            if (len(name_list)>3) and (len(n_list)>3):
                if n_list[:3] == name_list[:3]:
                    key = '_'.join(n_list[:3])
                    if not (key in dupes.keys()):
                        dupes[key] = [X_m[:62] + '_'.join(n_list) + X_m[-12:], X_m]
                    else:
                        dupes[key].append(X_m)
        names.append(name)
    return dupes


def box_plots_for_dupelicates():
    import matplotlib.pyplot as plt
    import numpy as np

    d_dict = load_data_dict()
    m_files = match_files(d_dict['2DUHRS_06_MIG_DEPTH'], d_dict['00_AI'])
    lines = []
    dupe_names = []
    dupe_labels = []
    dupe = find_duplicates(m_files=m_files)
    for key, val in dupe.items():
        trc = []
        lbls = []
        for file in val:
            trace, _ = get_traces(file, zrange=(25, 100))
            trc.append(trace.flatten())
            lbls.append(file[62:-12])
        lines.append(trc)
        dupe_names.append(key)
        dupe_labels.append(lbls)
    
    for name, labels, collection in zip(dupe_names, dupe_labels, lines):
        plt.clf()
        plt.boxplot(collection, labels=labels)
        plt.title(name)
        plt.xticks(rotation=10)
        plt.savefig('Data/dupelicates/{}.png'.format(name))


def img_plots_for_dupelicates():
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    d_dict = load_data_dict()
    m_files = match_files(d_dict['2DUHRS_06_MIG_DEPTH'], d_dict['00_AI'])
    lines = []
    dupe_names = []
    dupe_labels = []
    dupe = find_duplicates(m_files=m_files)
    for key, val in dupe.items():
        trc = []
        lbls = []
        for file in val:
            trace, _ = get_traces(file, zrange=(25, 100))
            trc.append(trace.T)
            lbls.append(file[62:-12])
        lines.append(trc)
        dupe_names.append(key)
        dupe_labels.append(lbls)
    
    for name, labels, collection in zip(dupe_names, dupe_labels, lines):
        plt.clf()
        fig, ax = plt.subplots(len(labels))
        fig.tight_layout(h_pad=1)
        for i, im in enumerate(collection):
            norm = Normalize(-2, 2)
            ax[i].imshow(im, cmap = 'seismic', norm=norm)
            ax[i].set_title(label=labels[i], fontsize=10)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        plt.savefig('Data/dupelicates/Image_{}.png'.format(name))


def negative_ai_values():
    import matplotlib.pyplot as plt
    import numpy as np
    from random import randint
    d_dict = load_data_dict()
    m_files = match_files(d_dict['2DUHRS_06_MIG_DEPTH'], d_dict['00_AI'])
    low_val = 0
    # y_below_zero = []
    t_b_z = []
    for file, i in m_files:
        # _, z = get_traces(file, zrange=(25, 100))
        traces, z_ai = get_traces(i, zrange=(25, 100))
        t = traces[np.where(np.any((traces<low_val), axis=1)), :]
        t_b_z+=list(t[0])
        # y_below_zero.append(t_b_z)
    # plt.hist(y_below_zero)
    print(np.shape(t_b_z))
    r = randint(0, len(t_b_z))
    plt.plot(t_b_z[r], z_ai)
    print('Minimum on the plot is: {}'.format(np.min(t_b_z[r])))
    print('Total minimum is {}'.format(np.min(t_b_z)))
    plt.show()

# Only used for testing the code
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # import numpy as np
    # from random import randint
    # d_dict = load_data_dict()
    # m_files = match_files(d_dict['2DUHRS_06_MIG_DEPTH'], d_dict['00_AI'])
    # z_high = []
    # y_below_zero = []
    # t_b_z = []
    # for file, i in m_files:
        # _, z = get_traces(file, zrange=(25, 100))
        # traces, z_ai = get_traces(i, zrange=(25, 100))
        # r = randint(0, len(traces))
        # plt.plot(traces[r], z_ai[::-1])
        # flat_t = traces.flatten()
        # t = traces[np.where(np.any((traces<-10000), axis=1)), :]
        # print(np.shape(t))
        # t_b_z+=list(t[0])
        
        #y_below_zero.append(list(t_b_z))

    # plt.hist(y_below_zero)
    # print(np.shape(t_b_z))
    # r = randint(0, len(t_b_z))
    # plt.plot(t_b_z[r], z_ai)
    # print('Minimum on the plot is: {}'.format(np.min(t_b_z[r])))
    # print('Total minimum is {}'.format(np.min(t_b_z)))
    # plt.show()

    # plt.boxplot(lines)
    # plt.show()
    # sgy_to_keras_dataset(['2DUHRS_06_MIG_DEPTH'], ['00_AI'], fraction_data=0.05, group_traces=3, normalize='StandardScaler')

    # import numpy as np
    # import matplotlib.pyplot as plt
    # from matplotlib.colors import Normalize
    # Display seismic traces at cpt locations
    # traces, z = match_cpt_to_seismic(n_neighboring_traces=500)
    # plt.imshow(traces.T, cmap='Greys', norm=Normalize(-3, 3))
    # plt.yticks(np.arange(0, 1400, 200), z[::200])
    # plt.show()

    from Architectures import CNN_collapsing_encoder, Collapse_CNN
    import keras
    from lightgbm import LGBMRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.manifold import TSNE

    X, y = create_sequence_dataset()

    A, b = create_sequence_dataset(ran=422)
    

    latent_model = CNN_collapsing_encoder(latent_features=16, image_width=11)

    # n_models = 1

    # print(latent_model.output)

    # models = [keras.Sequential(
    #     [
    #         keras.layers.Conv2D(16, (1, 1), activation="relu", padding='valid'),
    #         keras.layers.Conv2D(32, (1, 1), activation="relu", padding='valid'),
    #         keras.layers.Conv2D(64, (1, 1), activation="relu", padding='valid'),
    #         keras.layers.Conv2D(3, (1, 1), activation="relu")
    #         # keras.layers.Reshape((bs.shape[1], 3))
    #     ]
    #  )(latent_model.output) for _ in range(n_models)]

    # model = keras.Model(inputs=latent_model.input, outputs=models)


    # model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    # model.summary()

    # print(latent_model.predict(X).shape)

    # model = Collapse_CNN(latent_features=16, image_width=11, n_members=1)

    ann_decoder = keras.models.Sequential([
             keras.layers.Conv1D(16, 1, activation='relu', padding='same'),
             keras.layers.Conv1D(3, 1, activation='relu', padding='same')]
        )(latent_model.output)
    
    model = keras.Model(inputs=latent_model.input, outputs=ann_decoder)

    model.compile(loss='mse', optimizer='adam', metrics=['mse'])


    model.fit(X, y, epochs=400, batch_size=10, verbose=1)
    

    LGBM_model = MultiOutputRegressor(LGBMRegressor())


    # Fitting the LGBM model to the output of the latent model



    # Plot TSNE of the latent model
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(latent_model.predict(A[0].reshape(1, *A[0].shape)).reshape(b[0].shape[0], 16))

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        ax[i].scatter(tsne_results[:, 0], tsne_results[:, 1], c=b[0][:, i])
        ax[i].set_title('Feature {}'.format(i+1))



    plt.show()

    # Make three scatterplots of prediction results of the LGBM model
    # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # for i in range(3):
    #     ax[i].scatter(latent_model.predict(seis[:, :, :2*bs.shape[1]]).reshape(bs.shape[1], 16)[:, i], bs[0][:, i])
    #     ax[i].set_xlabel('Predicted')
    #     ax[i].set_ylabel('True')
    #     ax[i].invert_yaxis()
    #     ax[i].set_title('Feature {}'.format(i+1))
    # plt.show()

