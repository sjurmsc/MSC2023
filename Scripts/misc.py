import matplotlib.pyplot as plt

def find_all_relevant_soil_units():
    import segyio
    import numpy as np

    filename = '../OneDrive - NGI/Documents/NTNU/MSC_DATA/StructuralModel.sgy'
    with segyio.open(filename, ignore_geometry=True) as segyfile:
        z = segyfile.samples
        traces = segyio.collect(segyfile.trace)
        cdps = segyio.collect(segyfile.attributes(segyio.TraceField.CDP))
        lines = segyio.collect(segyfile.attributes(segyio.TraceField.INLINE_3D))
    
    print('Done reading the seismic')

    # Find all unique values in the seismic
    unique_values = np.unique(traces.T[np.where(z<100)])
    print('The unique soil units in the upper 100m are: ', unique_values)
    print(traces.shape)

    print(np.unique(lines))
    plt.plot(cdps)
    plt.show()


def illustrate_seq_lengths(n_neighboring_traces=5, 
                            zrange: tuple = (30, 100), 
                            n_bootstraps=20,
                            max_distance_to_cdp=25, # in meters (largest value in dataset is 21)
                            cumulative_seismic=False,
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
    Z = []

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
            cpt_vals = scaler.transform(cpt_vals)

        bootstraps, z_GM = bootstrap_CPT_by_seis_depth(cpt_vals, array(value['z_cpt']), z_GM, n=n_bootstraps, plot=False)

        seismic = array(value['Seismic_data'])
        if not seismic.shape[0] == image_width:
            print('Seismic sequences must conform with image width: {}'.format(key))
            continue

        for bootstrap in bootstraps:
            # Split CPT data by nan values
            row_w_nan = lambda x: np.any(np.isnan(x))
            nan_idx = np.apply_along_axis(row_w_nan, axis=1, arr=bootstrap)
            split_indices = np.unique([(i+1)*b for i, b in enumerate(nan_idx)])[1:]
            splits = np.split(bootstrap, split_indices)
            splits_depth = np.split(z_GM, split_indices)
            for s, sz in zip(splits, splits_depth):
                if np.all(np.isnan(s)): continue
                if (s.shape[0] > 1) and (s.shape[0] < 100):
                    y.append(s.shape[0])
                    Z.append(sz)

    data = {'y': y, 'Z': Z}

    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2))
    ax.hist(data, bins=20, color=msc_color, edgecolor='k', stacked=True, density=True)
    ax.set_xlabel('Length of CPT sequence', fontsize=8)
    ax.set_ylabel('Relative frequency', fontsize=8)
    fig.suptitle('CPT sequence lengths', fontsize=12)
    fig.tight_layout
    fig.subplots_adjust(bottom=0.236, left=0.279)
    fig.savefig('./Assignment Figures/Sequence_lengths.png', dpi=500)
    plt.show()



if __name__ == '__main__':
    from Feat_aug import *
    from Architectures import *
    from Log import *
    from pathlib import Path

    img_dir = './Assignment Figures/Depth_model_4/'

    if not Path(img_dir).exists():
        Path(img_dir).mkdir(parents=True)

    n_members = 1
    image_width = 11
    encoder_type = 'cnn'
    decoder_type = 'cnn'
    zrange = (30, 100)
    
    z = np.arange(zrange[0], zrange[1], 0.1).reshape(-1, 1)

    # normalize z between 0 and 1
    z = (z - z.min()) / (z.max() - z.min())
    
    x = np.arange(0, len(z), 1).reshape(-1, 1)
    
    x2 = x**2; x2 = x2.reshape(-1, 1)
    
    lx = np.log(z).reshape(-1, 1)
    
    
    
    
    full_args = create_full_trace_dataset(zrange=zrange, n_bootstraps=1)
    X_full = full_args[0]
    # Z_full = full_args[3]
    Z_full = np.array([z for i in range(X_full.shape[0])])


    args = create_sequence_dataset(zrange=zrange, n_bootstraps=1)
    X = args[0]

    # Normalize Z between 0 and 1
    Z = args[3]
    Z = (Z - Z.min()) / (Z.max() - Z.min())


    #x = X[0].reshape(1, 11, -1)

    # Z = np.array([z for i in range(X.shape[0])])
    # XS = np.array([x for i in range(X.shape[0])])
    # X2 = np.array([x2 for i in range(X.shape[0])])
    # LX = np.array([lx for i in range(X.shape[0])])

   

    # A = np.stack((Z, XS, X2, LX), axis=2)
    

    latent_features = 16

    encoder = CNN_pyramidal_encoder(latent_features=latent_features, image_width=image_width)
    
    decoder = keras.Sequential([
        keras.layers.InputLayer(input_shape=(None, latent_features)),
        keras.layers.Dense(32),
        keras.layers.Dense(64),
        keras.layers.Dense(1)
    ])

    # encoder = keras.models.load_model('depth_model_encoder_2.h5')

    model = keras.Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs))

    model.compile(optimizer='adam', loss='mae', metrics=['mse', 'mae'])
    
    model.fit(X, Z, epochs=10000, batch_size=1, verbose=1)

    encoder.save('depth_model_encoder_Z.h5')
    model.save('depth_model_Z.h5')

    # model = keras.models.load_model('depth_model_1.h5')
    # encoder = keras.models.load_model('depth_model_encoder_1.h5')

    Z_pred = model.predict(X)[:, :, 0]

    fig, ax = plt.subplots(1, 1, figsize=(5, 10))
    ax.plot(Z[0], z, label='True')
    
    for i in range(len(Z_pred)):
        # Plot the true and predicted depth along the y axis
        ax.plot(Z_pred[i], z, color=msc_color, alpha = 0.1, label='Predicted' if i==0 else None)

    # Flip x and y axis
    ax.invert_yaxis()

    # Set the x and y axis labels
    ax.set_xlabel('Predicted depth (m)')
    ax.set_ylabel('True depth (m)')

    ax.legend()
    
    fig.suptitle('Depth prediction')

    fig.savefig(img_dir + 'depth_prediction.png', dpi=500)

    # Create tsne plot of latent space colored by z
    from sklearn.manifold import TSNE

    # Get the latent space
    latent_space = encoder.predict(X)

    # Reshape the latent space to 2D
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    latent_space_2d = tsne.fit_transform(latent_space.reshape(-1, latent_features))

    # Plot the latent space
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(latent_space_2d[:, 0], latent_space_2d[:, 1], c=Z.flatten(), cmap='viridis')

    fig.savefig(img_dir + 'latent_space.png', dpi=500)

    # Plot the latent space colored by the predicted z
    create_latent_space_prediction_images(encoder, img_dir=img_dir, neighbors=600)

    # illustrate_seq_lengths(n_bootstraps=10, max_distance_to_cdp=25)
