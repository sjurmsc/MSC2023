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

    cnn_encoder = keras.Sequential([
        keras.layers.InputLayer(input_shape=image_shape),
        keras.layers.ZeroPadding2D(padding=((0, 0), (1, 1))),
        keras.layers.Conv2D(16, (3, 3), activation='relu'),
        keras.layers.ZeroPadding2D(padding=((0, 0), (1, 1))), # 1, 1 padding because kernel is 3x3
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((1, 2)), # Reduce the depth of seismic to GM_len
        keras.layers.Dropout(0.01)
    ], name='cnn_encoder')

    # Add more layers for shape reduction
    for _ in range((image_width-2*(3-1))//2):
        cnn_encoder.add(keras.layers.ZeroPadding2D(padding=((0, 0), (1, 1))))
        cnn_encoder.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        cnn_encoder.add(keras.layers.Dropout(0.01))


    cnn_encoder.add(keras.layers.Conv2D(latent_features, (1), activation='relu'))
    cnn_encoder.add(keras.layers.Reshape((-1, latent_features))) # Reshape to get features in the second dimension

    return cnn_encoder

def LSTM_encoder(latent_features, image_width):
    """LSTM encoder collapsing the dimension first dimension down to 1.
    Made to predict features at centered trace from seismic data."""
    assert image_width % 2 == 1, 'width % 2 != 1'

    image_shape = (image_width, None, 1) # None, because length is variable, 1 because monochromatic seismic

    lstm_encoder = keras.Sequential([
        keras.layers.InputLayer(input_shape=image_shape),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.LSTM(32, return_sequences=True),
        keras.layers.LSTM(16, return_sequences=True),
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


def ensemble_CNN_model(n_members=5, latent_features=16, image_width=11, learning_rate=0.001, enc='cnn', dec='cnn'):
    # 
    if enc == 'cnn':
        encoder = CNN_pyramidal_encoder(latent_features=latent_features, image_width=image_width)
    elif enc == 'lstm':
        encoder = LSTM_encoder(latent_features=latent_features, image_width=image_width)
   
    decoders = []
    for i in range(n_members):
        if dec == 'cnn':
            decoders.append(CNN_decoder(latent_features=latent_features, i=i)(encoder.output))
        elif dec == 'lstm':
            decoders.append(LSTM_decoder(latent_features=latent_features, i=i)(encoder.output))
    
    model_mean = Model(encoder.input, keras.layers.Average()(decoders))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model = Model(encoder.input, decoders)
    model.compile(loss='mae', optimizer=optimizer, metrics=['mse', 'mae'])

    model_mean.compile(loss='mae', optimizer=optimizer, metrics=['mse', 'mae'])

    return model, encoder, model_mean

def predict_encoded_tree(encoder, tree, X): #, mask=None):
    """Predicts the target variable from encoded data using a tree based
    multi attribute regressor."""

    encoded = encoder(X).numpy()
    encoded = encoded.reshape(-1, encoded.shape[-1])
    pred = tree.predict(encoded).reshape(X.shape[0], -1, 3)
    return pred
