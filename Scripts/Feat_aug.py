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
from os import listdir
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pickle
from sklearn.manifold import TSNE
# from JR.Seismic_interp_ToolBox import ai_to_reflectivity, reflectivity_to_ai

# Functions for loading data

def get_data_loc():
    d_filepath = './Data/data.json'
    return json.loads(d_filepath)

def get_cpt_name(cpt_loc):
    if cpt_loc < 86:
        cpt_name = f'TNW{cpt_loc:03d}'
    elif (cpt_loc>85) and (cpt_loc<=89):
        cpt_name = 'TNWTT{}'.format(int(cpt_loc % 85))
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

def match_cpt_to_seismic(n_neighboring_traces=0, zrange: tuple = (30, 100), to_file=''):
    distances = r'..\OneDrive - NGI\Documents\NTNU\MSC_DATA\Distances_to_2Dlines_Revised.xlsx'
    CPT_match = read_excel(distances)

    cpt_dict = get_cpt_las_files()

    match_dict = {}


    for i, row in CPT_match.iterrows():
        cpt_loc = int(row['Location no.'])

        # Get CPT name
        if cpt_loc < 86:
            cpt_name = f'TNW{cpt_loc:03d}'
        elif (cpt_loc>85) and (cpt_loc<=89):
            cpt_name = 'TNWTT{}'.format(int(cpt_loc % 85))

        sys.stdout.write('\rRetrieving from {}'.format(cpt_name))

        CDP = int(row['CDP'])
        seis_file = '../OneDrive - NGI/Documents/NTNU/MSC_DATA/2DUHRS_06_MIG_DEPTH/{}.sgy'.format(row['2D UHR line'])
        with segyio.open(seis_file, ignore_geometry=True) as SEISMIC:
            z = SEISMIC.samples
            a = array(SEISMIC.attributes(segyio.TraceField.CDP)).astype(int)
            traces = segyio.collect(SEISMIC.trace)[where(abs(a - CDP) <= n_neighboring_traces)]
            traces, z_traces = traces[:, (z>=zrange[0])&(z<zrange[1])], z[(z>=zrange[0])&(z<zrange[1])]
        
        CPT_DATA = cpt_dict[cpt_name].values
        CPT_DEPTH = cpt_dict[cpt_name].index.values

        # TEMPORARY
        SEAFLOOR = float(row['Water Depth'])/1000 # Convert from mm to m
        z_cpt = CPT_DEPTH + SEAFLOOR

        match_dict[row['Borehole']] = {'CDP'            : CDP, 
                                       'cpt_loc'        : cpt_loc,
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
                            add_noise=False,
                            cumulative_seismic=False,
                            random_flip=False, 
                            groupby='cpt_loc'):
    """
    Creates a dataset with sections of seismic image and corresponding CPT data where
    none of the CPT data is missing.
    """
    match_file = './Data/match_dict{}_z_{}-{}.pkl'.format(n_neighboring_traces, zrange[0], zrange[1])
    if not Path(match_file).exists():
        print('Creating match file...')
        match_dict = match_cpt_to_seismic(n_neighboring_traces, zrange, to_file=match_file)
    else:
        with open(match_file, 'rb') as file:
            match_dict = pickle.load(file)
    
    X, y = [], []
    groups = []

    print('Bootstrapping CPT data...')
    for key, value in match_dict.items():
        z_GM = np.arange(0, max(value['z_cpt']), 0.1)
        cpt_vals = array(value['CPT_data'])
        bootstraps, z_GM = bootstrap_CPT_by_seis_depth(cpt_vals, array(value['z_cpt']), z_GM, n=n_bootstraps, plot=False)

        correlated_cpt_z = z_GM # + value['seafloor']

        seismic = array(value['Seismic_data'])

        # if add_noise:
        #     seismic += np.random.normal(0, add_noise, seismic.shape)

        if cumulative_seismic:
            seismic = np.cumsum(seismic, axis=1)

        seismic_z = array(value['z_traces'])

        for bootstrap in bootstraps:
            # Split CPT data by nan values
            row_w_nan = lambda x: np.any(np.isnan(x))
            nan_idx = np.apply_along_axis(row_w_nan, axis=1, arr=bootstrap)
            split_indices = np.unique([(i+1)*b for i, b in enumerate(nan_idx)])[1:]
            splits = np.split(bootstrap, split_indices)
            splits_depth = np.split(correlated_cpt_z, split_indices)

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
                        X.append(X_val)
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


    return X, y, groups


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
            fig.savefig('./Assignment Figures/Latent_units/Latent_space_units_{}.png'.format(cpt_loc), dpi=1000)
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
            fig.savefig('./Assignment Figures/Latent_units/Latent_space_unit_{}.png'.format(i+1), dpi=1000)
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
    
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    
    fig.tight_layout()
    for i in range(3):
        ax[i].scatter(cpt_z, cpt_data[i], 'k', label='CPT data', s=1, alpha=0.5)
        ax[i].fill_betweenx(z, cpt_min[i], cpt_max[i], color='gray', alpha=0.5)
        ax[i].plot(z, cpt_mean[i],'k', linestyle='--')
        ax[i].plot(z, pred, 'r')

    if save:
        # Create Interpolation directory if it doesn't exist
        if not Path('./Assignment Figures/Interpolation').exists():
            Path('./Assignment Figures/Interpolation').mkdir()
        fig.savefig('./Assignment Figures/Interpolation/Interpolation_{}.png'.format(cpt), dpi=1000)
    
    else:
        plt.show()

    plt.close()


def plot_latent_space(latent_model, cpt_data, z_GM):
    """Use t-SNE to plot the latent space"""

    # The indices where no rows of y_val contain nan values
    valid_indices = np.where(np.all(~np.isnan(cpt_data), axis=1))[0]

    # The indexes of z where the depth matches valid indices of y_val
    valid_z_indices = np.where(np.isin(z, z_GM[valid_indices]))[0]
    outside_indices = np.where(~np.isin(np.arange(len(z)), valid_z_indices))[0]

    y_comp = cpt_data[np.where(np.all(~np.isnan(cpt_data), axis=1))[0]]

    # Plot TSNE of the latent model
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    prediction = latent_model.predict(X_val.reshape(1, *X_val.shape)).reshape(X_val.shape[1]//2, 16)
    tsne_results = tsne.fit_transform(prediction)

    
    cpt_parameters = ['$q_c$', '$f_s$', '$u_2$'] 
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        # Give specific markers to points outside the valid indices
        ax[i].scatter(tsne_results[outside_indices, 0], tsne_results[outside_indices, 1], marker='x', c='k', alpha=0.5)
        
        # Plot the valid indices
        ax[i].scatter(tsne_results[valid_z_indices, 0], tsne_results[valid_z_indices, 1], c=y_comp[:, i])

        ax[i].set_title('Latent space colored by {}'.format(cpt_parameters[i]))

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

    bin_min = np.array([])
    bin_max = np.array([])
    bin_mean = np.array([])
    for i, b in enumerate(cpt_bins):
        if not b.size: 
            bin_min = np.append(bin_min, np.nan)
            bin_max = np.append(bin_max, np.nan)
            bin_mean = np.append(bin_mean, np.nan)
        else: 
            bin_min = np.append(bin_min, np.min(b, axis=1))
            bin_max = np.append(bin_max, np.max(b, axis=1))
            bin_mean = np.append(bin_mean, np.mean(b, axis=1))

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


def box_plots_for_duplicates():
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



# Only used for testing the code
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # print('Load CPT locations')
    # df_CPT_loc=pd.DataFrame()


    # for path_las_files in ['P:/2019/07/20190798/Calculations/LasFiles/LAS_from_AGS_20210518', 'P:/2019/07/20190798/Calculations/LasFiles/LAS_from_AGS_Tennet_20210518']:
    #     for filename in os.listdir(path_las_files):
    #         if filename.endswith(".las" or ".LAS"):
    #             LAS = lasio.read(os.path.join(path_las_files, filename))
    #             tmpdict={'borehole':filename[:-4], 
    #                     'x':float(LAS.header['Well']['LOCX'].descr), 
    #                     'y':float(LAS.header['Well']['LOCY'].descr),
    #                     'TotalDepth':np.ceil(float(LAS.header['Well']['STOP'].value))
    #                     }
    #             df_CPT_loc=df_CPT_loc.append(tmpdict, ignore_index=True)


    # print('CPT locations loaded\n')


    # print('Load Water Depth')
    # df_CPT_loc=get_bathy_at_CPT(df_CPT_loc)
    # print('Water Depth loaded\n')

    # # Round WaterDepth to 0.02 m (CPT dz) to match dz from resampled seismic data and convert to mm
    # df_CPT_loc['WD'] = (1000*round(df_CPT_loc['WD']*50)/50).astype(int)


    # print('Assign CPT unique location ID')      ####### SPECIFIC TNW ##########
    # for index, row in df_CPT_loc.iterrows(): 
    #     if 'TT' in row['borehole']:
    #         ID = 85 + int(''.join(filter(lambda i: i.isdigit(), row['borehole'])))
    #     else:
    #         ID = int(''.join(filter(lambda i: i.isdigit(), row['borehole'])))
    #     df_CPT_loc.loc[index, 'ID'] = ID
    # print('CPT unique location ID assigned \n')

    # del index, row, ID, tmpdict, LAS, path_las_files, filename

    # df_CPT_loc=df_CPT_loc[['borehole','x','y','WD','TotalDepth','ID']]

    # dirpath_sgy = 'P:/2019/07/20190798/Background/2DUHRS_06_MIG_DEPTH/'
    # dirpath_NAV = 'P:/2019/07/20190798/Background/2DUHRS_06_MIG_DEPTH/nav_2DUHRS_06_MIG_DPT/'

    # df_NAV = load_NAV(dirpath_NAV)

    # get_seis_at_cpt_locations(df_NAV, dirpath_sgy, df_CPT_loc, n_tr=1)


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


    # import numpy as np
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
    

    from Architectures import CNN_collapsing_encoder #, Collapse_CNN
    import keras
    from lightgbm import LGBMRegressor
    from sklearn.multioutput import MultiOutputRegressor

    from sklearn.model_selection import train_test_split

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

    print('\nLoading model...')
    model = keras.models.load_model('model.h5')
    latent_model = keras.models.load_model('latent_model.h5')

    # LGBM_model = MultiOutputRegressor(LGBMRegressor())


    # Fitting the LGBM model to the output of the latent model
    # LGBM_model.fit(latent_model.predict(X_train), y_train)

    m_dict_path = r'.\Data\match_dict5_z_30-100.pkl'

    with open(m_dict_path, 'rb') as f:
        match_dict = pickle.load(f)

    X_val = match_dict['TNW054-PCPT']['Seismic_data']
    y_val = match_dict['TNW054-PCPT']['CPT_data']
    z_cpt = match_dict['TNW054-PCPT']['z_cpt']
    z_val = match_dict['TNW054-PCPT']['z_traces']
    z = match_dict['TNW054-PCPT']['z_traces'][::2]

    bootstraps, z_GM = bootstrap_CPT_by_seis_depth(y_val, z_cpt, z, n=1)
    bootstrap = bootstraps[0]

    # The indices where no rows of y_val contain nan values
    valid_indices = np.where(np.all(~np.isnan(bootstrap), axis=1))[0]

    # The indexes of z where the depth matches valid indices of y_val
    valid_z_indices = np.where(np.isin(z, z_GM[valid_indices]))[0]
    outside_indices = np.where(~np.isin(np.arange(len(z)), valid_z_indices))[0]

    y_comp = bootstrap[np.where(np.all(~np.isnan(bootstrap), axis=1))[0]]

    # Plot TSNE of the latent model
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    prediction = latent_model.predict(X_val.reshape(1, *X_val.shape)).reshape(X_val.shape[1]//2, 16)
    tsne_results = tsne.fit_transform(prediction)

    
    cpt_parameters = ['$q_c$', '$f_s$', '$u_2$'] 
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        # Give specific markers to points outside the valid indices
        ax[i].scatter(tsne_results[outside_indices, 0], tsne_results[outside_indices, 1], marker='x', c='k', alpha=0.5)
        
        # Plot the valid indices
        ax[i].scatter(tsne_results[valid_z_indices, 0], tsne_results[valid_z_indices, 1], c=y_comp[:, i])

        ax[i].set_title('Latent space colored by {}'.format(cpt_parameters[i]))

    plt.show()
    plt.close()

    create_latent_space_prediction_images(latent_model, neighbors=500)

    # # plot prediction results of the LGBM model in the same plot as a ground truth trace
    # X_val, y_val = match_dict['TNW054-PCPT']['Seismic_data'], match_dict['TNW054-PCPT']['CPT_data']
    # z_val = match_dict['TNW054-PCPT']['z_traces'][::2]

    # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # y_pred = LGBM_model.predict(latent_model.predict(X_val.reshape(1, *X_val.shape)).reshape(y_val.shape[0], 16))

    # for i in range(3):
    #     ax[i].plot(y_pred[:, i], z_val)
    #     ax[i].plot(y_val[:, i], z_val)
    #     ax[i].set_title('Feature {}'.format(i+1))
    #     ax[i].invert_yaxis()
    
    # plt.show()



    # Make three scatterplots of prediction results of the LGBM model
    # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # for i in range(3):
    #     ax[i].scatter(latent_model.predict(seis[:, :, :2*bs.shape[1]]).reshape(bs.shape[1], 16)[:, i], bs[0][:, i])
    #     ax[i].set_xlabel('Predicted')
    #     ax[i].set_ylabel('True')
    #     ax[i].invert_yaxis()
    #     ax[i].set_title('Feature {}'.format(i+1))
    # plt.show()

