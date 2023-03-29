"""
Functions to be used for feature augmentation
"""
import segyio
import json
from numpy import array, row_stack, intersect1d, where, amax, amin, stack
from pandas import read_csv
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle
from os import listdir
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pickle
from sklearn.manifold import TSNE
import numpy as np
from Architectures import predict_encoded_tree
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
# import sns


# from JR.Seismic_interp_ToolBox import ai_to_reflectivity, reflectivity_to_ai

# Functions for loading data

def get_data_loc():
    d_filepath = './Data/data.json'
    return json.loads(d_filepath)

def get_cpt_name(cpt_loc):
    if cpt_loc < 86:
        cpt_name = f'TNW{cpt_loc:03d}'
    elif (cpt_loc>85) and (cpt_loc<=89):
        cpt_name = 'TNWTT{:02d}'.format(int(cpt_loc % 85))
    return cpt_name

def get_cpt_data_from_file(fp, zrange: tuple = (30, 100)):
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

def match_cpt_to_seismic(n_neighboring_traces=0, zrange: tuple = (30, 100), to_file='', data_folder = 'FE_CPT'):

    CPT_match = read_excel(r'..\OneDrive - NGI\Documents\NTNU\MSC_DATA\Distances_to_2Dlines_Revised.xlsx')

    cpt_dict = get_cpt_las_files(cpt_folder_loc = '../OneDrive - NGI/Documents/NTNU/MSC_DATA/{}'.format(data_folder))


    match_dict = {}


    for i, row in CPT_match.iterrows():
        

        # if not row['Borehole'] in cpt_dict.keys():
        #     continue

        cpt_loc = int(row['Location no.'])

        # Get CPT name
        cpt_name = get_cpt_name(cpt_loc)

        # Combined is joined at cpt location
        if data_folder == 'combined':
            cpt_key = cpt_name
        elif data_folder == 'FE_CPT':
            cpt_key = row['Borehole']

        distance = row['Distance to CDP']

        



        sys.stdout.write('\rRetrieving from {} \t\t'.format(cpt_key))

        CDP = int(row['CDP'])
        seis_file = '../OneDrive - NGI/Documents/NTNU/MSC_DATA/2DUHRS_06_MIG_DEPTH/{}.sgy'.format(row['2D UHR line'])
        with segyio.open(seis_file, ignore_geometry=True) as SEISMIC:
            z = SEISMIC.samples
            a = array(SEISMIC.attributes(segyio.TraceField.CDP)).astype(int)
            traces = segyio.collect(SEISMIC.trace)[where(abs(a - CDP) <= n_neighboring_traces)]
            traces, z_traces = traces[:, (z>=zrange[0])&(z<zrange[1])], z[(z>=zrange[0])&(z<zrange[1])]
        
        try: CPT_DATA = cpt_dict[cpt_key].values
        except KeyError: print('CPT key not found: {}'.format(cpt_key)); continue
        
        CPT_DEPTH = cpt_dict[cpt_key].index.values

        # TEMPORARY
        SEAFLOOR = float(row['Water Depth'])/1000 # Convert from mm to m
        z_cpt = CPT_DEPTH + SEAFLOOR

        match_dict[row['Borehole']] = {'CDP'            : CDP,
                                       'distance'       : distance, 
                                       'cpt_loc'        : cpt_loc,
                                       'Borehole'       : row['Borehole'],
                                       'CPT_data'       : CPT_DATA, 
                                       'Seismic_data'   : traces, 
                                       'z_traces'       : z_traces, 
                                       'z_cpt'          : z_cpt,
                                       'seafloor'       : SEAFLOOR}
    sys.stdout.flush()
    sys.stdout.write('\nDone!\n')
    

    if to_file:
        with open(to_file, 'wb') as file:
            pickle.dump(match_dict, file)
        sys.stdout.write('Saved to file: {}\n'.format(to_file))

    sys.stdout.flush()
    return match_dict
        

def create_sequence_dataset(n_neighboring_traces=5, 
                            zrange: tuple = (30, 100), 
                            n_bootstraps=20,
                            sequence_length=10, 
                            stride=1,
                            max_distance_to_cdp=25, # in meters (largest value in dataset is 21)
                            add_noise=False,
                            cumulative_seismic=False,
                            random_flip=False,
                            random_state=0,
                            groupby='cpt_loc',
                            data_folder = 'FE_CPT',
                            y_scaler=None):
    """
    Creates a dataset with sections of seismic image and corresponding CPT data where
    none of the CPT data is missing.
    """
    match_file = './Data/match_dict{}_z_{}-{}_ds_{}.pkl'.format(n_neighboring_traces, zrange[0], zrange[1], data_folder)
    if not Path(match_file).exists():
        print('Creating match file...')
        match_dict = match_cpt_to_seismic(n_neighboring_traces, zrange, to_file=match_file)
    else:
        with open(match_file, 'rb') as file:
            match_dict = pickle.load(file)
    
    image_width = 2*n_neighboring_traces + 1

    X, y = [], []
    groups = []

    if y_scaler is not None:
        scaler = get_cpt_data_scaler()

    print('Bootstrapping sequence CPT data...')
    for key, value in match_dict.items():

        # Skip if distance to CDP is too large
        if abs(value['distance']) > max_distance_to_cdp:
            continue

        z_GM = np.arange(zrange[0], zrange[1], 0.1)
        cpt_vals = array(value['CPT_data'])

        if y_scaler is not None:
            cpt_vals = y_scaler.transform(cpt_vals)

        bootstraps, z_GM = bootstrap_CPT_by_seis_depth(cpt_vals, array(value['z_cpt']), z_GM, n=n_bootstraps, plot=False)

        seismic = array(value['Seismic_data'])
        if not seismic.shape[0] == image_width:
            print('Seismic sequences must conform with image width: {}'.format(key))
            continue

        if cumulative_seismic:
            seismic = np.cumsum(seismic, axis=1)

        seismic_z = array(value['z_traces'])

        for bootstrap in bootstraps:
            # Split CPT data by nan values
            row_w_nan = lambda x: np.any(np.isnan(x))
            nan_idx = np.apply_along_axis(row_w_nan, axis=1, arr=bootstrap)
            split_indices = np.unique([(i+1)*b for i, b in enumerate(nan_idx)])[1:]
            splits = np.split(bootstrap, split_indices)
            splits_depth = np.split(z_GM, split_indices)

            for section, z in zip(splits, splits_depth):
                if (section.shape[0]-1) > sequence_length:
                    in_seis = np.where((seismic_z >= z.min()-1e-6) & (seismic_z < z.max()-1e-6)) # -1e-6 to avoid floating point errors
                    seis_seq = seismic[:, in_seis][:, 0, :]
                    cpt_seq = section[:-1, :]
                    if not seis_seq.shape[1] == 2*cpt_seq.shape[0]:
                        print('Seismic sequences must represent the same interval as the CPT: {}'.format(key))
                        continue

                    # Split the sequence into multiple overlapping sequences of length sequence_length

                    for i, j in zip(range(0, cpt_seq.shape[0]-sequence_length, stride), range(0, seis_seq.shape[1]-2*sequence_length, 2*stride)):
                        X_val = seis_seq[:, j:j+2*sequence_length]
                        if add_noise:
                            if cumulative_seismic:
                                X_val += np.cumsum(np.random.normal(0, add_noise, X_val.shape), axis=1)
                            else:
                                X_val += np.random.normal(0, add_noise, X_val.shape)
                        
                        X.append(X_val.reshape(X_val.shape[0], X_val.shape[1], 1))
                        y_val = cpt_seq[i:i+sequence_length, :]
                        y.append(y_val)
                        if groupby == 'cpt_loc':
                            groups.append(int(value['cpt_loc']))
                        elif groupby == 'borehole':
                            groups.append(int(key))

    
    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)   

    # Randomly flip the X data about the 1 axis
    if random_flip:
        for i in range(len(X)):
            if np.random.randint(2):
                X[i] = np.flip(X[i], 0)

    if random_state:
        X, y, groups = shuffle(X, y, groups, random_state=random_state)

    print('Done!')

    return X, y, groups


def create_full_trace_dataset(n_neighboring_traces=5,
                              zrange: tuple = (30, 100), 
                              n_bootstraps=5,
                              max_distance_to_cdp=25, # in meters (largest value in dataset is 21)
                              add_noise=False,
                              cumulative_seismic=False,
                              random_flip=False,
                              random_state=0,
                              groupby='cpt_loc',
                              data_folder='FE_CPT',
                              force_new_match_file=False,
                              y_scaler=None,
                              ydata='bootstraps'):
    """ Creates a dataset with full seismic traces and corresponding CPT data with an array
        containing the indices where there are nan values in a row and one where are no nan values in a row.
    """
    match_file = './Data/match_dict{}_z_{}-{}_ds_{}.pkl'.format(n_neighboring_traces, zrange[0], zrange[1], data_folder)
    if (not Path(match_file).exists()) or force_new_match_file:
        print('Creating match file...')
        match_dict = match_cpt_to_seismic(n_neighboring_traces, zrange, to_file=match_file, data_folder=data_folder)
    else:
        with open(match_file, 'rb') as file:
            match_dict = pickle.load(file)
    
    X, y = [], []
    nan_idxs = []
    no_nan_idxs = []
    sw_idxs = []
    extrapolated_idxs = []
    groups = []
    GGM = []
    mins = []
    maxs = []

    if y_scaler is not None:
        scaler = get_cpt_data_scaler()


    print('Bootstrapping Full trace CPT data...')
    for key, value in match_dict.items():
    
        # Skip if distance to CDP is too large
        if abs(value['distance']) > max_distance_to_cdp:
            continue

        z_min, z_max = zrange[0], zrange[1]

        z_GM = np.arange(z_min, z_max, 0.1)
        cpt_vals = array(value['CPT_data'])

        image_width = 2*n_neighboring_traces+1

        # Normalize CPT data
        if y_scaler is not None:
            cpt_vals = scaler.transform(cpt_vals)

        if ydata == 'bootstraps':
            bootstraps, z_GM = bootstrap_CPT_by_seis_depth(cpt_vals, array(value['z_cpt']), z_GM, n=n_bootstraps)
        elif ydata == 'mmm':
            b_min, b_max, b_mean, z_GM = get_max_min_and_mean_for_depth_bins(cpt_vals, array(value['z_cpt']), z_GM)

        seismic = array(value['Seismic_data'])
        if not seismic.shape[0] == image_width:
            print('Seismic image must conform with image width: {}'.format(key))
            continue

        if cumulative_seismic:
            seismic = np.cumsum(seismic, axis=1)

        seismic_z = array(value['z_traces'])
        seismic_z = seismic_z[where((seismic_z >= z_min) & (seismic_z < z_max))]
        cpt_z = z_GM[where((z_GM >= z_min) & (z_GM < z_max))]

        

        # Indices for sea water
        sf = value['seafloor']
        is_sw = lambda z: z<sf
        sw_idx = np.apply_along_axis(is_sw, axis=0, arr=z_GM)
        
        # Assign GGM to the trace
        ggm = assign_ggm_to_picks(value['Borehole'])
        ggm[sw_idx] = 0

        if ydata == 'bootstraps':
            for bootstrap in bootstraps:
                # List the indices of the CPT data that are not nan
                row_w_nan = lambda x: np.any(np.isnan(x))
                nan_idx = np.apply_along_axis(row_w_nan, axis=1, arr=bootstrap)
                nan_idxs.append(nan_idx)

                row_wo_nan = lambda x: np.all(~np.isnan(x))
                no_nan_idx = np.apply_along_axis(row_wo_nan, axis=1, arr=bootstrap)
                no_nan_idxs.append(no_nan_idx)

                # Adding sea water indices
                sw_idxs.append(sw_idx)

                # Adding GGM
                GGM.append(ggm)
                # Adding extrapolated indices
                # extrapolated_idx = np.zeros(bootstrap.shape[0], dtype=bool)
                # last = np.where(~nan_idx)[0][-1]
                # extrapolated_idx[last:] = True

                # extrapolated_idxs.append(extrapolated_idx)

                seis = seismic.copy()
                if add_noise:
                    if cumulative_seismic:
                        seis += np.cumsum(np.random.normal(0, add_noise, seis.shape), axis=1)
                    else:
                        seis += np.random.normal(0, add_noise, seis.shape)

                if seis.shape[1] != 2*bootstrap.shape[0]:
                    print(seis.shape[1], 2*bootstrap.shape[0])
                    print('Seismic and CPT data do not match in length: {}'.format(key))
                    continue
                
                X.append(seis.reshape(seis.shape[0], seis.shape[1], 1))
                y.append(bootstrap)

                # Adding the group value
                if groupby == 'cpt_loc':
                    groups.append(int(value['cpt_loc']))
                elif groupby == 'borehole':
                    raise ValueError('Grouping by borehole is not supported for full trace dataset')
        elif ydata == 'mmm':
            # List the indices of the CPT data that are not nan
            row_w_nan = lambda x: np.any(np.isnan(x))
            nan_idx = np.apply_along_axis(row_w_nan, arr=b_mean, axis=1)
            nan_idxs.append(nan_idx)

            row_wo_nan = lambda x: np.all(~np.isnan(x))
            no_nan_idx = np.apply_along_axis(row_wo_nan, axis=1, arr=b_mean)
            no_nan_idxs.append(no_nan_idx)

            # Adding sea water indices
            sw_idxs.append(sw_idx)

            # Adding GGM
            GGM.append(ggm)
            # Adding extrapolated indices
            # extrapolated_idx = np.zeros(bootstrap.shape[0], dtype=bool)
            # last = np.where(~nan_idx)[0][-1]
            # extrapolated_idx[last:] = True

            # extrapolated_idxs.append(extrapolated_idx)

            mins.append(b_min)
            maxs.append(b_max)

            seis = seismic.copy()
            if add_noise:
                if cumulative_seismic:
                    seis += np.cumsum(np.random.normal(0, add_noise, seis.shape), axis=1)
                else:
                    seis += np.random.normal(0, add_noise, seis.shape)

            if seis.shape[1] != 2*b_mean.shape[0]:
                print(seis.shape[1], 2*b_mean.shape[0])
                print('Seismic and CPT data do not match in length: {}'.format(key))
                continue
            
            X.append(seis.reshape(seis.shape[0], seis.shape[1], 1))
            y.append(b_mean)

            # Adding the group value
            if groupby == 'cpt_loc':
                groups.append(int(value['cpt_loc']))
            elif groupby == 'borehole':
                raise ValueError('Grouping by borehole is not supported for full trace dataset')    
    
    X = np.array(X)
    y = np.array(y)
    nan_idxs = np.array(nan_idxs)
    no_nan_idxs = np.array(no_nan_idxs)
    sw_idxs = np.array(sw_idxs)
    groups = np.array(groups)
    mins = np.array(mins)
    maxs = np.array(maxs)

    # Randomly flip the X data about the 1 axis
    if random_flip:
        for i in range(len(X)):
            if np.random.randint(2):
                X[i] = np.flip(X[i], 0)
    
    if random_state:
        X, y, groups, nan_idxs, no_nan_idxs = shuffle(X, y, groups, nan_idxs, no_nan_idxs, random_state=random_state)

    print('Done!')

    if ydata == 'mmm':
        return X, y, groups, nan_idxs, no_nan_idxs, sw_idxs, extrapolated_idxs, GGM, (mins, maxs)

    return X, y, groups, nan_idxs, no_nan_idxs, sw_idxs, extrapolated_idxs, GGM
            

def get_struct_model_picks(line_name, CDP):
    """Get the structural model from dat file"""

    pick_path = Path('../OneDrive - NGI/Documents/NTNU/MSC_DATA/02_Picks/')

    picks = pick_path.glob('*.dat')

    # line_names = np.unique([l.stem for l in Path(r"C:\Users\SjB\OneDrive - NGI\Documents\NTNU\MSC_DATA\2DUHRS_06_MIG_DEPTH").glob('*.sgy')])

    if line_name[-4:] == '_DPT':
        line_name = line_name[:-4]

    horizons = {}

    for pick in picks:

        h = pick.stem.split('_')[2]
        # df = pd.read_csv(pick, sep='\t')

        with open(pick, 'r') as readfile:
            it = [l.split() for l in readfile.readlines()]
            cols = ['Profile', 'CDP', 'Easting (m)', 'Northing (m)', 'TWT (ms)', 'Depth (mLAT)']
            df = pd.DataFrame(it[1:], columns=cols)

        try: pick_depth = df.loc[(df['Profile'] == line_name) & (df['CDP'] == str(CDP))]['Depth (mLAT)'].values.astype(float)[0]
        except: continue

        horizons[h] = pick_depth
    
    return horizons


def create_pick_dict(to_file=''):
    """Create a dictionary with the picks for each line and CDP"""
    CPT_locs = pd.read_excel(r'..\OneDrive - NGI\Documents\NTNU\MSC_DATA\Distances_to_2Dlines_Revised.xlsx')

    picks = {}
    for i, row in CPT_locs.iterrows():
        line = row['2D UHR line']
        CDP = row['CDP']
        horizons = get_struct_model_picks(line, CDP)
        picks[row['Borehole']] = horizons
    
    if to_file:
        with open(to_file, 'wb') as f:
            pickle.dump(picks, f)
    
    return picks


def assign_ggm_to_picks(bh, zrange=(30, 100)):
    """Assign the GGM to the picks"""
    unit_mapping = read_csv(r'..\OneDrive - NGI\Documents\NTNU\MSC_DATA\StructuralModel_unit_mapping.csv', index_col=0)

    # Load pick dict
    pick_dict_filename = './Data/pick_dict.pkl'

    if not Path(pick_dict_filename).exists():
        pick_dict = create_pick_dict(to_file=pick_dict_filename)
    else:
        with open(pick_dict_filename, 'rb') as f:
            pick_dict = pickle.load(f)

    z = np.arange(zrange[0], zrange[1], 0.1)
    GGM = np.ones_like(z)

    picks = pick_dict[bh]
    
    pick_list = picks.items()
    pick_list = sorted(pick_list, key=lambda x: x[1])

    for i, pick in enumerate(pick_list):
        if i == 0:
            lower = 'R00'
        else:
            lower = pick_list[i-1][0]
        
        higher = pick_list[i][0]

        GGM[z < pick[1]] = pick_2_GGM(lower, higher)
    
    return GGM



def pick_2_GGM(lower, higher):
    """Assign a GGM to a pick
       lower and higher refers to values of z"""
    if (lower == 'R00') and (higher == 'R01'):
        return 1
    elif ((lower == 'R01') or (lower == 'R00')) and ((higher == 'R02') or (higher == 'R10')):
        return 2
    elif (lower == 'R02') and (higher == 'R20'):
        return 3
    elif (lower == 'R10') and (higher == 'R20'):
        return 4
    elif (lower == 'R20') and (higher == 'R21'):
        return 5
    elif ((lower == 'R20') or (lower == 'R21')) and (higher == 'R22'):
        return 6
    elif (lower == 'R22') and ((higher == 'R23') or (higher == 'R30')):
        return 7
    elif (lower == 'R23') and (higher == 'R30'):
        return 8
    elif (lower == 'R30') and (higher == 'R31'):
        return 10 # 10 and 11 are joined together
    elif ((lower == 'R30') or (lower == 'R31')) and (higher == 'R32'):
        return 12
    elif ((lower == 'R31') or (lower == 'R32')) and (higher == 'R34'):
        return 16
    elif (lower == 'R40') and (higher == 'R41'):
        return 17
    elif ((lower == 'R40') or (lower == 'R41')) and (higher == 'R50'):
        return 18
    elif (lower == 'R50') and (higher == 'R51'):
        return 19
    elif ((lower == 'R50') or (lower == 'R51')) and (higher == 'R52'):
        return 20
    elif ((lower == 'R50') or (lower == 'R52')) and (higher == 'R53'):
        return 21
    elif ((lower == 'R50') or (lower == 'R53')) and (higher == 'R60'):
        return 22
    elif (lower == 'R60'):
        return 23
    else:
        return -1




#### Image creation functions ####

def create_latent_space_prediction_images(model, oob='', neighbors = 200, image_width = 11, groupby='cpt_loc'):
    distances = r'..\OneDrive - NGI\Documents\NTNU\MSC_DATA\Distances_to_2Dlines_old.xlsx'
    CPT_match = read_excel(distances)

    img_neighbors = int((image_width-1)/2)

    if groupby == 'latent_unit':
        pred_image = array([]).reshape((16, len(CPT_match['Location no.'].unique()), neighbors*2+1, seis.shape[0]))

    for i, row in CPT_match.iterrows():
        cpt_loc = row['Location no.']
        CDP = row['CDP']
        seis_file = '../OneDrive - NGI/Documents/NTNU/MSC_DATA/2DUHRS_06_MIG_DEPTH/{}.sgy'.format(row['2D UHR line'])
        with segyio.open(seis_file, ignore_geometry=True) as f:
            seis = f.trace.raw[:]
            seis = seis.reshape((seis.shape[0], seis.shape[1], 1))
            
            CDP_index = np.where(array(f.attributes(segyio.TraceField.CDP)) == CDP)[0][0]
            
        pred_image = array([]).reshape((0, seis.shape[1]//2, 16))
        
        img_left = CDP_index-neighbors
        img_right = min([CDP_index+neighbors, seis.shape[0]-(img_neighbors+1)]) # CDP 79 is close to the edge of the image

        for ii in range(img_left, img_right+1):
            sys.stdout.write('\rPredicting image {} of {} ({}%)'.format(i+1, len(CPT_match), int((ii-img_left)/(img_right-img_left+1)*100)))
            sys.stdout.flush()
            l_loc = ii-img_neighbors
            r_loc = ii+img_neighbors+1
            latent_pred = model.predict(seis[l_loc:r_loc, :, 0].reshape((1, image_width, seis.shape[1])), verbose=0)[0]
            pred_image = row_stack((pred_image, latent_pred))
        
        if groupby == 'latent_unit':
            pred = pred_image.reshape((16, 1, neighbors*2+1, seis.shape[0]))
            if i == 0:
                unit_pred = np.array(pred).reshape((16, 0, neighbors*2+1, seis.shape[0]))
            else:
                unit_pred = np.concatenate((unit_pred, pred), axis=1)
            

        if groupby == 'cpt_loc':
            fig, ax = plt.subplots(2, 8, figsize=(20, 5))
            fig.tight_layout()
            fig.subplots_adjust(top=0.85)

            for ii in range(16):
                ax[ii//8, ii%8].imshow(seis[img_left:img_right+1, :2:-1, 0 ].T, cmap='gray')
                ax[ii//8, ii%8].imshow(pred_image[:, :, ii].T, cmap='gist_rainbow', alpha=0.4)
                ax[ii//8, ii%8].axis('off')
                ax[ii//8, ii%8].set_title('Latent {}'.format(ii+1))
            fig.suptitle('Latent space prediction for CPT location {}'.format(cpt_loc))
            fig.savefig('./Assignment Figures/Latent_units/Latent_space_units_{}.png'.format(cpt_loc), dpi=500)
            plt.close()
            print('\nSaved image for CPT location {}'.format(cpt_loc))

    if groupby == 'latent_unit':

        for i in range(16):
            preds = unit_pred[i, :, :, :]
            fig, ax = plt.subplots(9, 11, figsize=(20, 15))
            fig.tight_layout()
            fig.subplots_adjust(top=0.85)
            for ii in range(89):
                ax[ii//11, ii%11].imshow(seis[img_left:img_right+1, :2:-1, 0 ].T, cmap='gray')
                ax[ii//11, ii%11].imshow(preds[ii, :, :].T, cmap='gist_rainbow', alpha=0.4)
                ax[ii//11, ii%11].axis('off')
                ax[ii//11, ii%11].set_title('CPT {}'.format(CPT_match['Location no.'].unique()[ii]))
            fig.suptitle('Latent space prediction for latent unit {}'.format(i+1))
            fig.savefig('./Assignment Figures/Latent_units/Latent_space_unit_{}.png'.format(i+1), dpi=500)
            plt.close()
            print('\nSaved image for latent unit {}'.format(i+1))


def create_sgy_of_latent_predictions(model, seismic_dir, image_width = 11):
    """Create a SEGY file with the latent space predictions for each CDP location"""
    seismic_lines = Path(seismic_dir).glob('*.sgy')

    for line in seismic_lines:
        with segyio.open(line, 'r') as f:
            seis = f.trace.raw[:]
            seis = seis.reshape((seis.shape[0], seis.shape[1], 1))
            CDPs = array(f.attributes(segyio.TraceField.CDP))
            CDPs = CDPs.reshape((CDPs.shape[0], 1))
            print('Creating latent space predictions for {}'.format(line.name))
            pred_image = array([]).reshape((0, seis.shape[1]//2, 16))
            img_neighbors = int((image_width-1)/2)
            for ii in range(seis.shape[0]):
                sys.stdout.write('\rPredicting image {} of {} ({}%)'.format(ii+1, seis.shape[0], int(ii/seis.shape[0]*100)))
                sys.stdout.flush()
                l_loc = ii-img_neighbors
                r_loc = ii+img_neighbors+1
                latent_pred = model.predict(seis[l_loc:r_loc, :, 0].reshape((1, image_width, seis.shape[1])))[0]
                pred_image = row_stack((pred_image, latent_pred))

            # Create a file for every unit of the latent space
            for i in range(16):
                with segyio.create('./OpendTect/'+line.name[:-4]+'_latent_unit{}.sgy'.format(i+1), spec=f.spec) as dst:
                    dst.bin = f.bin
                    dst.header = f.header
                    dst.trace = pred_image[:, :, i]
                    dst.attributes(segyio.TraceField.CDP)[:] = CDPs


def plot_interp_in_discontinuous_cpt(model, image_width = 11):

    print('Plotting predictions for discontinuous CPTs')

    discon_cpts = [7, 11, 16, 30, 34, 38, 39, 41, 42, 48, 49, 54, 69, 76]

    for cpt in discon_cpts:
        plot_cpt_pred(model, cpt, save=True, image_width=image_width)


def plot_cpt_pred(model, cpt, save = False, image_width = 11):

    cpt_dir = r'C:/Users/SjB/OneDrive - NGI/Documents/NTNU/MSC_DATA/combined/'

    distances = r'..\OneDrive - NGI\Documents\NTNU\MSC_DATA\Distances_to_2Dlines_old.xlsx'
    CPT_match = read_excel(distances)

    name = get_cpt_name(cpt)
    cpt_filename = cpt_dir + name + '.las'
    cpt_data, cpt_z = get_cpt_data_from_file(cpt_filename)

    row = CPT_match.loc[CPT_match['Location no.'] == cpt]
    line = row['Line'].values[0]
    seis_file = Path(r'..\OneDrive - NGI\Documents\NTNU\MSC_DATA\Seismic\Seismic_2D').glob(line+'*.sgy')
    cdp = row['CDP'].values[0]

    with segyio.open(seis_file, 'r') as f:
        seis = f.trace.raw[where(abs(f.attributes(segyio.TraceField.CDP)-cdp)<(image_width-1)//2)[0][0]]
    pred = model.predict(seis.reshape((1, image_width, seis.shape[1])))[0]
    cpt_min, cpt_max, cpt_mean, z = get_max_min_and_mean_for_depth_bins(cpt_data=cpt_data, cpt_depth=cpt_z, 
                                                                        GM_depth=np.arange(cpt_z.min(), cpt_z.max(), 0.1))
    

    pred_color = ['g', 'orange', 'b']

    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    
    fig.tight_layout()
    for i in range(3):
        ax[i].scatter(cpt_z, cpt_data[i], 'k', label='CPT data', s=1, alpha=0.5)
        ax[i].fill_betweenx(z, cpt_min[i], cpt_max[i], color='gray', alpha=0.5)
        ax[i].plot(z, cpt_mean[i],'k', linestyle='--')
        ax[i].plot(z, pred, pred_color[i])

    if save:
        # Create Interpolation directory if it doesn't exist
        if not Path('./Assignment Figures/Interpolation').exists():
            Path('./Assignment Figures/Interpolation').mkdir()
        fig.savefig('./Assignment Figures/Interpolation/Interpolation_{}.png'.format(cpt), dpi=500)
    
    else:
        plt.show()

    plt.close()


def plot_latent_space(latent_model, X, valid_indices, outside_indices, GGM, filename=''):
    """Use t-SNE to plot the latent space"""

    # Plot TSNE of the latent model
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    prediction = latent_model.predict(X).reshape((-1, 16))
    tsne_results = tsne.fit_transform(prediction)

    outside_indices = outside_indices.flatten()
    valid_indices = valid_indices.flatten()
    GGM = GGM.flatten()
    
    umap = read_csv('../OneDrive - NGI/Documents/NTNU/MSC_DATA/StructuralModel_unit_mapping.csv')
    GGM_names = []
    for u in GGM:
        GGM_names.append(umap.loc[umap['uid'] == u]['unit'].values[0])

    n_colors = len(np.unique(umap['uid']))
    # Add a segmented colorbar with unique colors for the different units
    cmap = plt.cm.get_cmap('gnuplot', n_colors)

    bounds = np.arange(min(umap['uid']), max(umap['uid']), 1)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    # Set padding
    fig.subplots_adjust(left=0.05, top=0.9, right=0.85, bottom=0.1)

    # Give specific markers to points outside the valid indices
    ax.scatter(tsne_results[outside_indices, 0], tsne_results[outside_indices, 1], marker='x', c=GGM[outside_indices], cmap=cmap, norm=norm, alpha=0.8, label='Extrapolated GGM')
    
    # Plot the valid indices
    ax.scatter(tsne_results[valid_indices, 0], tsne_results[valid_indices, 1], marker= 'o', c=GGM[valid_indices], cmap=cmap, norm=norm, alpha=0.8, label='Validated GGM')

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Create a colorbar
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=np.arange(-0.5, n_colors-0.5, 1))

    cbar.ax.set_yticklabels(umap['unit'].unique())
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label('Ground model units', fontsize=20)

    fig.suptitle('Latent space colored by Ground model units', fontsize=20)

    # Insert a legend for the markers without color, and at alpha=1
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', fontsize=15)

    # Remove the frame of the figure
    for spine in ax.spines.values():
        spine.set_visible(False)
    


    if filename:
        fig.savefig(filename, dpi=500)
    else:
        plt.show()
    plt.close()


def create_loo_trace_prediction(model, test_X, test_y, zrange=(30, 100), filename='', title='', minmax=None):

    # Get scaler
    scaler = get_cpt_data_scaler()

    test_y = test_y.copy()
    mins = minmax[0].copy()
    maxs = minmax[1].copy()

    # Create predictions for the test set
    if not type(model) == list:
        predictions = model.predict(test_X)
    else:
        encoder, model = model
        predictions = predict_encoded_tree(encoder, model, test_X)

    # Rescale the predictions
    for i in range(predictions.shape[0]):
        predictions[i, :, :] = scaler.inverse_transform(predictions[i, :, :])
        test_y[i, :, :] = scaler.inverse_transform(test_y[i, :, :])
        mins[i, :, :] = scaler.inverse_transform(mins[i, :, :])
        maxs[i, :, :] = scaler.inverse_transform(maxs[i, :, :])
    

    z = np.arange(zrange[0], zrange[1], 0.1)

    units = ['$q_c$', '$f_s$', '$u_2$']
    ax_labels = ['Measured tip resistance [MPa]', 'Sleeve Friction [MPa]', 'Pore pressure [MPa]']
    pred_color = ['g', 'orange', 'b']

    # Create a figure for the predictions 
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.tight_layout()
    for i in range(3):
        for t in range(predictions.shape[0]):
            ax[i].plot(predictions[t, :, i], z, 'k', alpha=0.1)
            # Plot test_y using only markers
            ax[i].plot(test_y[t, :, i], z, pred_color[i], marker='.', alpha=0.5)
            if minmax is not None:
                ax[i].fill_betweenx(z, mins[t, :, i], maxs[t, :, i], color=pred_color[i], alpha=0.1)
        ax[i].set_title(units[i])
        ax[i].set_ylabel('Depth [mLAT]')

        # Set the x axis label to the top
        # ax[i].xaxis.set_label_position('top')
        ax[i].set_xlabel(ax_labels[i])

        ax[i].invert_yaxis()
    # Add super title
    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(top=0.85)

    if filename:
        fig.savefig(filename, dpi=500)
    else:
        plt.show()
    plt.close()


def prediction_scatter_plot(model, test_X, test_y, filename='', title=''):
    """Plot the predictions of the model as a scatter plot, with density contours"""
    
    # Get scaler
    scaler = get_cpt_data_scaler()

    test_y = test_y.copy()

    # Create predictions for the test set
    if not type(model) == list:
        predictions = model.predict(test_X)
    else:
        encoder, model = model
        predictions = predict_encoded_tree(encoder, model, test_X)

    # Rescale the predictions
    for i in range(predictions.shape[0]):
        predictions[i] = scaler.inverse_transform(predictions[i])
        test_y[i] = scaler.inverse_transform(test_y[i])

    units = ['$q_c$', '$f_s$', '$u_2$']
    pred_color = ['g', 'orange', 'b']

    # Create a figure for the predictions 
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    for i in range(3):
        p = predictions[:, :, i].flatten()
        t = test_y[:, :, i].flatten()
        # Plot predictions using only markers
        ax[0, i].scatter(t, p, c=pred_color[i], marker='.', alpha=0.5)

        # Plot a kdeplot with the predictions
        sns.kdeplot(x=t, y=p, ax=ax[0, i], cmap='Blues', fill=True, thresh=0.05, alpha=0.5)

        ax[0, i].set_title(units[i])

        # Add the 1:1 line
        ax[0, i].plot([0, 1], [0, 1], transform=ax[0, i].transAxes, ls='--', c='k')

        # Add axis labels
        ax[0, i].set_xlabel('True {} [MPa]'.format(units[i]))
        ax[0, i].set_ylabel('Predicted {} [MPa]'.format(units[i]))

        # Plot the histogram of the residuals
        ax[1, i].hist((p-t), bins=50, edgecolor='k')
        ax[1, i].set_xlabel('Residuals')
        ax[1, i].set_ylabel('Frequency')

        
    # Add super title
    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(top=0.9, wspace=0.3, hspace=0.3)

    if filename:
        fig.savefig(filename, dpi=500)
    else:
        plt.show()
    plt.close()


# from NGI.GM_BuildDatabase import *
import scipy.spatial as spatial
def get_seis_at_cpt_locations(df_NAV, dirpath_sgy, df_CPT_loc=pd.DataFrame([]), n_tr = 1):
    # def get_seis_at_CPT(df_NAV, dirpath_sgy, df_CPT_loc=pd.DataFrame([]), n_tr = 1):
    ''' 
    Extract seismic trace(s) closest to CPT location and add to database
    df_NAV: dataframe with Navigation data
    dirpath_sgy: path to directory with corresponding SEGY files
    df_CPT_loc: dataframe with CPT locations
    n_tr: number of traces to extract
    '''
    # Quadtree decomposition of the Navigation
    xy = df_NAV[['x','y']]
    print('\nQuadtree decomposition ...\n')
    tree = spatial.KDTree(xy)

    xl_path = r'..\OneDrive - NGI\Documents\NTNU\MSC_DATA\Distances_to_2Dlines_Revised.xlsx'
    xl = read_excel(xl_path)

    # Loop over all CPT locations find closest location and add trace to database
    print('Merging seismic traces')
    df_seisall = pd.DataFrame()
    for ind, row in df_CPT_loc.iterrows():
        distance, index = tree.query(row[['x','y']], k=n_tr)
        if n_tr>1:
            indexes = index.flatten()
            distances = distance.flatten()
            linenames = df_NAV.iloc[indexes, df_NAV.columns.get_loc('line')]
            tracls = df_NAV.iloc[indexes, df_NAV.columns.get_loc('tracl')]
        else:
            indexes = index
            distances = [np.array(distance)]
            linenames = [df_NAV.iloc[indexes, df_NAV.columns.get_loc('line')]]
            tracls = [df_NAV.iloc[indexes, df_NAV.columns.get_loc('tracl')]]
            
        # Extract seismic trace using segyio
        ii = 0
        df_seisloc = pd.DataFrame()
        
        for line, tracl, dist in zip(linenames, tracls, distances):
            print('Line: {}, tracl: {}, distance: {}'.format(line, tracl, dist))
            ii = ii+1
            path_seis = dirpath_sgy + line + '.sgy'
            with segyio.open(path_seis, 'r', ignore_geometry=True) as f:
                # Get basic attributes
                sample_rate = segyio.tools.dt(f) # in mm
                n_samples = f.samples.size
                z_seis = f.samples*1000        # in mm


                # Extract CDP number of the trace
                cdp = f.attributes(segyio.TraceField.CDP)[tracl][0]
                print('CDP: {}, Tracl: {}'.format(cdp, tracl))

                # Extract CDP X and Y coordinates
                cdp_x , cdp_y = row[['x','y']]

                # Add row to excel with line, cdp, cdp_x, cdp_y, dist
                xl = xl.append({'Location no.': row['ID'],
                                'Borehole': row['borehole'], 
                                '2D UHR line': line, 
                                'CDP': cdp, 
                                'Location Eastings': cdp_x, 
                                'Location Northings': cdp_y, 
                                'Distance to CDP': dist,
                                'Water Depth': row['WD']}, ignore_index=True)
                

                # Load nearest trace(s)
                tr = f.trace[tracl]
        xl.sort_values(by=['Location no.'], inplace=True)
        xl.to_excel(xl_path, index=False)
        print('Saved to excel')
    print('seismic traces merged')
    return df_seisall
  

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


def get_max_min_and_mean_for_depth_bins(cpt_data, cpt_depth, GM_depth):
    """ This function creates bins of cpt values at ground model depths, 
        calculates the mean, max, and min which are returned."""
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

    bin_min = np.array([]).reshape(-1, 3)
    bin_max = np.array([]).reshape(-1, 3)
    bin_mean = np.array([]).reshape(-1, 3)
    for i, b in enumerate(cpt_bins):
        if not b.size: 
            bin_min = np.row_stack((bin_min, np.full((1, 3), np.nan)))
            bin_max = np.row_stack((bin_max, np.full((1, 3), np.nan)))
            bin_mean = np.row_stack((bin_mean, np.full((1, 3), np.nan)))
        else: 
            bin_min = np.row_stack((bin_min, np.min(b, axis=0)))
            bin_max = np.row_stack((bin_max, np.max(b, axis=0)))
            bin_mean = np.row_stack((bin_mean, np.nanmean(b, axis=0)))

    return bin_min, bin_max, bin_mean, GM_depth


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

    return df_dict


import pandas as pd
def get_csv_of_cdp_location_coordinates():
    seismic_folder = '../OneDrive - NGI/Documents/NTNU/MSC_DATA/2DUHRS_06_MIG_DEPTH/'
    seismic_files = [f for f in listdir(seismic_folder) if f.endswith('.sgy')]
    seismic_files.sort()
    line = []
    cdp = []
    x = []
    y = []
    for file in seismic_files:
        with segyio.open(seismic_folder + file, ignore_geometry=True) as f:
            cdp.append(f.attributes(segyio.TraceField.CDP_X)[0])
            x.append(f.attributes(segyio.TraceField.SourceX)[0])
            y.append(f.attributes(segyio.TraceField.SourceY)[0])
            line.append(file[0:-4])
    
    # Create a dataframe
    df = pd.DataFrame({'Line': line, 'CDP': cdp, 'X': x, 'Y': y})
    df.to_csv('../QGIS/CDP_location.csv', index=False)


from math import  gcd
def sequence_length_histogram(val='CPT'):
    """Creates a histogram plot of the length of the sequences in the dataset"""
    X, y = create_sequence_dataset()

    idx = 0
    if val == 'CPT':
        value = y
        
    elif val == 'Seismic':
        value = X
        idx = 1

    lengths = []
    for b in value:
        if b.shape[idx] < 50:
            lengths.append(b.shape[idx])

    print('Least common divisor for lengths are {}'.format(gcd(*lengths)))

    plt.hist(lengths, bins=30)
    plt.title('Length of {} sequences, under 100'.format(val))
    plt.show()


def get_cpt_data_scaler(t='minmax'):
    """Gets the scaler for the CPT data"""
    
    scale_fpath = './Data/Scaler/CPT_scaler_{}.pkl'.format(t)

    if not Path('./Data/Scaler/').exists():
        Path('./Data/Scaler/').mkdir(parents=True)

    if Path(scale_fpath).exists():
        with open(scale_fpath, 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    else:

        # Load CPT data
        cpt_dict = get_cpt_las_files()

        # Stack all data values
        cpt_data = array([]).reshape(0, 3)
        for _, val in cpt_dict.items():
            cpt_data = row_stack((cpt_data, val.values))
        cpt_data = array(cpt_data)


        if t == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError('Scaler type not supported')
        
        scaler.fit(cpt_data)

        # Save scaler
        with open(scale_fpath, 'wb') as f:
            
            pickle.dump(scaler, f)

        print('Saved scaler to {}'.format(scale_fpath))

        return scaler


# Only used for testing the code
if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # from matplotlib.colors import Normalize

    # # Display seismic traces at cpt locations

    # match_dict = match_cpt_to_seismic(n_neighboring_traces=500)
    # traces = match_dict[1]['Seismic_data']
    # z = match_dict[1]['z_traces']
    # plt.imshow(np.array(traces).T, cmap='Greys', norm=Normalize(-3, 3))
    # plt.yticks(np.arange(0, len(z), 200), z[::200])
    # plt.show()
    # match_dict = match_cpt_to_seismic(n_neighboring_traces=n_neighboring_traces)
    

    # import keras
    # from lightgbm import LGBMRegressor
    # from sklearn.multioutput import MultiOutputRegressor

    # from sklearn.model_selection import train_test_split

    # get_csv_of_cdp_location_coordinates()

    # sequence_length_histogram(val='CPT')

    image_width = 11
    n_neighboring_traces = (image_width-1)//2
    # X, y, groups = create_sequence_dataset(n_bootstraps=1, sequence_length=10)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    # latent_model = CNN_collapsing_encoder(latent_features=16, image_width=image_width)

    # n_models = 1

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

    # model = Collapse_CNN(latent_features=16, image_width=11, n_members=1)

    # ann_decoder = keras.models.Sequential([
    #             keras.layers.Conv1D(16, 1, activation='relu', padding='same'),
    #             keras.layers.Conv1D(32, (1, 1), activation="relu", padding='valid'),
    #             keras.layers.Conv1D(64, (1, 1), activation="relu", padding='valid'),
    #             keras.layers.Conv1D(3, 1, activation='relu', padding='same')]
    #         )(latent_model.output)
    
    # model = keras.Model(inputs=latent_model.input, outputs=ann_decoder)

    # model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    # model.summary()

    # history = model.fit(X_train, y_train, epochs=1, batch_size=10, verbose=1, validation_data=(X_val, y_val))

    # model.save('model.h5')
    # latent_model.save('latent_model.h5')

    # print('\nLoading model...')
    # model = keras.models.load_model('model.h5')
    # latent_model = keras.models.load_model('latent_model.h5')

    # LGBM_model = MultiOutputRegressor(LGBMRegressor())


    # Fitting the LGBM model to the output of the latent model
    # LGBM_model.fit(latent_model.predict(X_train), y_train)

    # m_dict_path = r'.\Data\match_dict5_z_30-100.pkl'

    # with open(m_dict_path, 'rb') as f:
    #     match_dict = pickle.load(f)

    # X_val = match_dict['TNW054-PCPT']['Seismic_data']
    # y_val = match_dict['TNW054-PCPT']['CPT_data']
    # z_cpt = match_dict['TNW054-PCPT']['z_cpt']
    # z_val = match_dict['TNW054-PCPT']['z_traces']
    # z = match_dict['TNW054-PCPT']['z_traces'][::2]

    # pick = get_struct_model_picks('TNW_B01_5460_MIG_DPT', 2095)


    # pick_dict_filename = './Data/pick_dict.pkl'

    # if not Path(pick_dict_filename).exists():
    #     pick_dict = create_pick_dict(to_file=pick_dict_filename)
    # else:
    #     with open(pick_dict_filename, 'rb') as f:
    #         pick_dict = pickle.load(f)
    

    # assing_ggm_to_picks(pick_dict)

    from keras.models import load_model

    full_trace = create_full_trace_dataset(n_bootstraps=1, n_neighboring_traces=n_neighboring_traces, y_scaler='minmax', ydata='mmm')

    X_full = full_trace[0]
    y_full = full_trace[1]
    no_nan = full_trace[4]
    nans = full_trace[3]
    sw = full_trace[5]
    GGM = full_trace[7]

    minmax = full_trace[-1]

    GGM = np.array(sw).astype(int)-1

    model_loc = r"C:\Users\SjB\MSC2023\Models\ALW\Fold1\Ensemble_CNN_encoder_0.h5"

    encoder = load_model(model_loc)
    model = load_model(model_loc.replace('_encoder', ''))

    idx = 0

    # plot_latent_space(encoder, X_full[idx].reshape(1, *X_full[0].shape), no_nan[idx], nans[idx], GGM[idx])
    create_loo_trace_prediction(model, X_full[idx].reshape(1, *X_full[0].shape), y_full[idx].reshape(1, *y_full[0].shape), minmax=minmax)
    prediction_scatter_plot(model, X_full[idx].reshape(1, *X_full[0].shape), y_full[idx].reshape(1, *y_full[0].shape), title='Fold 1, model 0, trace 0')