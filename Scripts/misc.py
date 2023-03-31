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

if __name__ == '__main__':
    from Feat_aug import *
    from Architectures import *
    from Log import *
    from pathlib import Path

    img_dir = './Assignment Figures/Depth_model/'

    if not Path(img_dir).exists():
        Path(img_dir).mkdir(parents=True)

    n_members = 1
    image_width = 11
    encoder_type = 'cnn'
    decoder_type = 'cnn'
    zrange = (30, 100)
    z = np.arange(zrange[0], zrange[1], 0.1)
    args = create_full_trace_dataset(zrange=zrange, n_bootstraps=1)

    X = args[0]
    x = X[0].reshape(1, 11, -1)

    Z = np.array([z for i in range(X.shape[0])])

    latent_features = 16

    encoder = CNN_pyramidal_encoder(latent_features=latent_features, image_width=image_width)
    
    decoder = keras.Sequential([
        keras.layers.InputLayer(input_shape=(None, latent_features)),
        # keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
        keras.layers.Dense(1)
    ])

    # encoder = keras.models.load_model('depth_model_encoder.h5')

    model = keras.Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs))

    model.compile(optimizer='adam', loss='mae', metrics=['mse', 'mae'])
    
    model.fit(X, Z, epochs=2, batch_size=1, verbose=1)

    encoder.save('depth_model_encoder.h5')
    Z_pred = model.predict(X)

    fig, ax = plt.subplots(1, 1, figsize=(5, 15))
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
    ax.scatter(latent_space_2d[:, 0], latent_space_2d[:, 1], c=Z[0], cmap='viridis')

    fig.savefig(img_dir + 'latent_space.png', dpi=500)

    # Plot the latent space colored by the predicted z
    create_latent_space_prediction_images(encoder, img_dir=img_dir)
