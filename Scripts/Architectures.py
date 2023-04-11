"""
Contains the model architectures so that they may be called by other scripts.
"""
import keras
from keras import Model
import tensorflow as tf

def CNN_pyramidal_encoder(latent_features, image_width):
    """2D CNN encoder collapsing the dimension first dimension down to 1.
    Made to predict features at centered trace from seismic data."""

    assert image_width % 2 == 1, 'width % 2 != 1'

    # input = keras.layers.Input(shape=(None, image_width, 1), ragged=True)

    image_shape = (image_width, None, 1) # None, because length is variable, 1 because monochromatic seismic

    # Input block
    inp = keras.layers.Input(shape=image_shape)
    x = keras.layers.GaussianNoise(0.01)(inp)
    im = keras.layers.RandomFlip(mode='vertical')(x)

    # First convolutional block
    x = keras.layers.ZeroPadding2D(padding=((0, 0), (2, 2)))(im)
    x = keras.layers.Conv2D(16, (5, 5), activation='relu')(x) # Reduce horizontal dimension by 2
    x = keras.layers.Conv2D(16, (5, 5), activation='relu', padding='same')(x)
    x = keras.layers.Conv2D(16, (5, 5), activation='relu', padding='same')(x)
    x = keras.layers.Conv2D(16, (5, 5), activation='relu', padding='same')(x)

    # Second convolutional block
    x = keras.layers.ZeroPadding2D(padding=((0, 0), (1, 1)))(x) # 1, 1 padding because kernel is 3x3
    x = keras.layers.SpatialDropout2D(0.1)(x)
    x = keras.layers.Conv2D(32, (3, 3), strides=(1, 2), activation='relu')(x) # Reduce horizontal dimension by 2, and vertical by factor 2
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    # Third convolutional block
    x = keras.layers.ZeroPadding2D(padding=((0, 0), (1, 1)))(x) # 2, 2 padding because kernel is 5x5
    x = keras.layers.SpatialDropout2D(0.1)(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x) # Reduce horizontal dimension by 4
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    # Add more layers for shape reduction
    x = keras.layers.ZeroPadding2D(padding=((0, 0), (1, 1)))(x) # 1, 1 padding because kernel is 3x3
    x = keras.layers.Conv2D(32, (3, 3), activation='relu')(x) # Reduce horizontal dimension by 2
    x = keras.layers.Conv2D(latent_features, (3, 3), activation='relu', padding='same')(x)
    outp = keras.layers.Reshape((-1, latent_features))(x) # Reshape to get features in the second dimension

    cnn_encoder = Model(inp, outp)

    return cnn_encoder



def pyramidal_residual_encoder(latent_features, image_width, nb_stacks=5, nb_filters=16, kernel_size=3, activation='relu'):
    """2D CNN encoder with skip connections collapsing the first dimension down to 1."""
    
    assert image_width % 2 == 1, 'width % 2 != 1'

    image_shape = (image_width, None, 1) # None, because length is variable, 1 because monochromatic seismic

    # Input layer
    input_layer = keras.layers.InputLayer(input_shape=image_shape)

    # First convolutional layer
    conv1 = keras.layers.Conv2D(nb_filters, kernel_size, activation=activation, padding='same', name='conv1_1')(input_layer.output)
    conv1 = keras.layers.Conv2D(nb_filters, kernel_size, activation=activation, padding='same', name='conv1_2')(conv1)
    

    # Stacked convolutional layers
    conv = conv1
    for i in range(nb_stacks):
        conv = keras.layers.Conv2D(nb_filters, kernel_size, activation=activation, padding='same', name='conv{}_1'.format(i+2))(conv)
        conv = keras.layers.Conv2D(nb_filters, kernel_size, activation=activation, padding='same', name='conv{}_2'.format(i+2))(conv)
        conv = keras.layers.SpatialDropout2D(0.1)(conv)
        conv = keras.layers.Conv2D(nb_filters, kernel_size, activation=activation, padding='same', name='conv{}_3'.format(i+2))(conv)
        conv = keras.layers.Conv2D(nb_filters, kernel_size, activation=activation, padding='same', name='conv{}_4'.format(i+2))(conv)
        conv = keras.layers.SpatialDropout2D(0.1)(conv)
        conv = keras.layers.Conv2D(nb_filters, kernel_size, activation=activation, padding='same', name='conv{}_5'.format(i+2))(conv)
        conv = keras.layers.Conv2D(nb_filters, kernel_size, activation=activation, padding='same', name='conv{}_6'.format(i+2))(conv)
    
    # Skip connections
    conv = keras.layers.Add(name='skip')([conv, conv1])

    pool1 = keras.layers.AveragePooling2D(pool_size=(11, 2), name='pool1')(conv)
    
    # Last convolutional layer
    conv = keras.layers.Conv2D(latent_features, kernel_size, activation=activation, padding='same', name='conv{}_1'.format(nb_stacks+2))(pool1)
    conv = keras.layers.Conv2D(latent_features, kernel_size, activation=activation, padding='same', name='conv{}_2'.format(nb_stacks+2))(conv)
    conv = keras.layers.Conv2D(latent_features, kernel_size, activation=activation, padding='same', name='conv{}_3'.format(nb_stacks+2))(conv)
    conv = keras.layers.Conv2D(latent_features, kernel_size, activation=activation, padding='same', name='conv{}_4'.format(nb_stacks+2))(conv)
    conv = keras.layers.Conv2D(latent_features, kernel_size, activation=activation, padding='same', name='conv{}_5'.format(nb_stacks+2))(conv)
    conv = keras.layers.Conv2D(latent_features, kernel_size, activation=activation, padding='same', name='conv{}_6'.format(nb_stacks+2))(conv)
    
    conv = keras.layers.Add(name='skip2')([conv, pool1])

    conv = keras.layers.Conv2D(latent_features, kernel_size, activation=activation, padding='same', name='conv{}_1'.format(nb_stacks+3))(conv)

    # Reshape to get features in the second dimension
    conv = keras.layers.Reshape((-1, latent_features))(conv)

    # Create model
    model = Model(inputs=input_layer.input, outputs=conv, name='pyramidal_residual_encoder')

    return model



def LSTM_encoder(latent_features, image_width, mask=None):
    """LSTM encoder collapsing the dimension first dimension down to 1.
    Made to predict features at centered trace from seismic data."""
    assert image_width % 2 == 1, 'width % 2 != 1'

    image_shape = (image_width, None, 1) # None, because length is variable, 1 because monochromatic seismic

    lstm_encoder = keras.Sequential([
        keras.layers.InputLayer(input_shape=image_shape),
        keras.layers.ConvLSTM1D(32, 3, activation='relu', padding='same', return_sequences=True),
        keras.layers.ConvLSTM1D(32, 3, activation='relu', padding='same', return_sequences=True),
        keras.layers.Conv2D(32, (3, 3), strides=(1, 2), activation='relu', padding='same'),
        keras.layers.ConvLSTM1D(16, 3, activation='relu', padding='same', return_sequences=True),
        keras.layers.ConvLSTM1D(latent_features, 3, activation='relu', padding='same'),
        keras.layers.Dense(latent_features)
    ], name='lstm_encoder')

    return lstm_encoder



def LSTM_decoder(latent_features=16, i=0):
    """LSTM decoder predicting CPT response from latent features."""
    lstm_decoder = keras.Sequential([
        keras.layers.InputLayer(input_shape=(None, latent_features)),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.LSTM(32, return_sequences=True),
        keras.layers.LSTM(16, return_sequences=True),
        keras.layers.Dense(3)
    ], name='lstm_decoder_{}'.format(i))

    return lstm_decoder


def CNN_decoder(latent_features=16, i=0):
    """1D CNN decoder with a committee of n_members."""
    cnn_decoder = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=(None, latent_features)),
        keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
        keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
        keras.layers.Conv1D(16, 3, activation='relu', padding='same'),
        keras.layers.Dense(3)
    ], name='cnn_decoder_{}'.format(i))

    return cnn_decoder

def ANN_decoder(latent_features=16, i=0):
    """1D CNN decoder with a committee of n_members."""
    ann_decoder = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=(None, latent_features)),
        # keras.layers.Dense(64, activation='relu'),
        # keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(3)
    ], name='ann_decoder_{}'.format(i))

    return ann_decoder

def ensemble_CNN_model(n_members=5, latent_features=16, image_width=11, learning_rate=0.001, enc='cnn', dec='cnn'):
    # 
    if enc == 'cnn':
        encoder = CNN_pyramidal_encoder(latent_features=latent_features, image_width=image_width)
        # encoder = pyramidal_residual_encoder(latent_features=latent_features, image_width=image_width)
    elif enc == 'lstm':
        encoder = LSTM_encoder(latent_features=latent_features, image_width=image_width)
    elif enc == 'depth':
        encoder = keras.models.load_model('depth_model_encoder_2.h5')
   
    # decoders = []
    for i in range(n_members):
        if dec == 'cnn':
            # decoders.append(CNN_decoder(latent_features=latent_features, i=i)(encoder.output))
            decoders = CNN_decoder(latent_features=latent_features, i=i)(encoder.output)
        elif dec == 'lstm':
            decoders.append(LSTM_decoder(latent_features=latent_features, i=i)(encoder.output))
        elif dec == 'ann':
            decoders.append(ANN_decoder(latent_features=latent_features, i=i)(encoder.output))
    
    

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model = Model(encoder.input, decoders)
    model.compile(loss='mae', optimizer=optimizer, metrics=['mse', 'mae'])


    # if len(decoders)>1:
    #     model_mean = Model(encoder.input, keras.layers.Average()(decoders))
    #     model_mean.compile(loss='mae', optimizer=optimizer, metrics=['mse', 'mae'])
    # else:
    model_mean = model

    return model, encoder, model_mean

def predict_encoded_tree(encoder, tree, X): #, mask=None):
    """Predicts the target variable from encoded data using a tree based
    multi attribute regressor."""

    encoded = encoder(X).numpy()
    encoded = encoded.reshape(-1, encoded.shape[-1])
    pred = tree.predict(encoded).reshape(X.shape[0], -1, 3)
    return pred
