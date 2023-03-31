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
    n_members = 1
    image_width = 11
    encoder_type = 'cnn'
    decoder_type = 'cnn'
    zrange = (30, 100)
    z = np.arange(zrange[0], zrange[1], 0.1)
    args = create_full_trace_dataset(zrange=zrange)

    X = args[0]
    x = X[0].reshape(1, 11, -1)

    Z = np.array([z for i in range(X.shape[0])])

    encoder = CNN_pyramidal_encoder(latent_features=4, image_width=image_width)
    
    decoder = keras.Sequential([
        keras.layers.InputLayer(input_shape=(None, 4)),
        # keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
        keras.layers.Dense(1)
    ])
    
    model = keras.Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs))

    model.compile(optimizer='adam', loss='mae', metrics=['mse', 'mae'])
    
    model.fit(X, Z, epochs=10000, batch_size=1, verbose=1)

    encoder.save('depth_model_encoder.h5')

    Z_pred = model.predict(X)

    fig, ax = plt.subplots(1, 1, figsize=(10, 15))
    for i in range(10):
        # Plot the true and predicted depth along the y axis
        ax.plot(Z[i+1], np.arange(len(Z[0])), label='True')
        ax.plot(Z_pred[i+1], np.arange(len(Z[0])), label='Pred')


    
    # Flip x and y axis
    ax.invert_yaxis()

    plt.legend()
    plt.show()
