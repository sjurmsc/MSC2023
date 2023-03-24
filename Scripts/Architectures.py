"""
Contains the model architectures so that they may be called by other scripts.
"""
import keras
from keras import Model, Input
from keras.layers import Layer

# from tensorflow_addons.layers import WeightNormalization
# from tensorflow_addons.metrics import RSquare
from numpy import array
import tensorflow as tf
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

class TCN_encoder(Model):
    def __init__(self, **kwargs):
        super(TCN_encoder, self).__init__(**kwargs)
        self.tcn = TCN(nb_stacks=5,
                       nb_filters=16, 
                       kernel_size=(3, 9), 
                       dilations=[1, 2, 4, 8, 16, 32, 64], 
                       padding='same',
                       reduce_rows=True,
                       return_sequences=True, 
                       use_skip_connections=True, 
                       dropout_rate=0.2,
                       activation='relu', 
                       use_batch_norm=True, 
                       use_layer_norm=False, 
                       use_weight_norm=False, 
                       name='tcn_encoder')

    def build(self, input_shape):
        self.tcn.build(input_shape)
        self.tcn.summary()
        super(TCN_encoder, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return self.tcn.compute_output_shape(input_shape)

    def call(self, input):
        return self.tcn(input)


class LGBM_decoder(Model):
    def __init__(self, **kwargs):
        super(LGBM_decoder, self).__init__(**kwargs)
        self.lgbm = LGBMRegressor()

    def call(self, input):
        return self.lgbm.predict(input)


class TCN_enc_LGBM_dec(Model):
    """TCN encoder + LGBM decoder model.
    Made to predict CPT from seismic data."""


    def __init__(self, tcn_encoder):
        self.tcn_encoder = tcn_encoder
        self.lgbm_decoder = LGBMRegressor()

    def compile(self):
        self.tcn_encoder.compile(optimizer='adam', loss='mse')

    def train_step(self, batch_data):
        X, y = batch_data

        latent_space = self.tcn_encoder(X)
        self.lgbm_decoder.fit(latent_space, y)

        with tf.GradientTape() as tape:
            loss = 1. - self.lgbm_decoder.score(latent_space, y) # 1 - because score is R^2
        
        gradients = tape.gradient(loss, self.tcn_encoder.trainable_variables)
        self.tcn_encoder.optimizer.apply_gradients(zip(gradients, self.tcn_encoder.trainable_variables))

        return loss

    def predict(self, X):
        return self.lgbm_decoder.predict(self.tcn_encoder(X))

    def score(self, X, y):
        return self.lgbm_decoder.score(self.tcn_encoder(X), y)

    def __repr__(self):
        return 'TCN_enc_LGBM_dec(tcn_encoder={}, lgbm_decoder={})'.format(self.tcn_encoder, self.lgbm_decoder)

    def __str__(self):
        return 'TCN_enc_LGBM_dec(tcn_encoder={}, lgbm_decoder={})'.format(self.tcn_encoder, self.lgbm_decoder)
    

def compiled_tcn_enc_lgbm_dec(trainig_data, **kwargs):
    X, y = trainig_data
    tcn_encoder = TCN_encoder()
    tcn_enc_lgbm_dec = TCN_enc_LGBM_dec(tcn_encoder)
    tcn_enc_lgbm_dec.compile()
    tcn_enc_lgbm_dec.fit(X=X, y=y, **kwargs)
    return tcn_enc_lgbm_dec


class TCN_enc_RF_dec(Model):
    """TCN encoder + Random Forest decoder model.
    Made to predict CPT from seismic data."""


    def __init__(self, tcn_encoder):
        self.tcn_encoder = tcn_encoder
        self.rf_decoder = RandomForestRegressor()

    def compile(self):
        self.tcn_encoder.compile(optimizer='adam', loss='mse')

    def train_step(self, batch_data):
        X, y = batch_data

        latent_space = self.tcn_encoder(X)
        self.rf_decoder.fit(latent_space, y)

        with tf.GradientTape() as tape:
            loss = 1. - self.rf_decoder.score(latent_space, y) # 1 - because score is R^2
        
        gradients = tape.gradient(loss, self.tcn_encoder.trainable_variables)
        self.tcn_encoder.optimizer.apply_gradients(zip(gradients, self.tcn_encoder.trainable_variables))

        return loss

    def predict(self, X):
        return self.rf_decoder.predict(self.tcn_encoder(X))

    def score(self, X, y):
        return self.rf_decoder.score(self.tcn_encoder(X), y)

    def __repr__(self):
        return 'TCN_enc_RF_dec(tcn_encoder={}, rf_decoder={})'.format(self.tcn_encoder, self.rf_decoder)

    def __str__(self):
        return 'TCN_enc_RF_dec(tcn_encoder={}, rf_decoder={})'.format(self.tcn_encoder, self.rf_decoder)
    

def compiled_tcn_enc_rf_dec(trainig_data, **kwargs):
    X, y = trainig_data
    tcn_encoder = TCN_encoder()
    tcn_enc_rf_dec = TCN_enc_RF_dec(tcn_encoder)
    tcn_enc_rf_dec.compile()
    tcn_enc_rf_dec.fit(X=X, y=y, **kwargs)
    return tcn_enc_rf_dec


class ANN_committee(Layer):
    """ANN committee layer.
    Made to predict CPT from seismic data."""


    def __init__(self, n_members=5, **kwargs):
        super(ANN_committee, self).__init__(**kwargs)

        self.opc = keras.Sequential(
        [
            keras.layers.Conv2D(16, (1, 1), activation="relu", padding='valid'),
            keras.layers.Conv2D(32, (1, 1), activation="relu", padding='valid'),
            keras.layers.Conv2D(64, (1, 1), activation="relu", padding='valid'),
            keras.layers.Conv2D(3, (1, 1), activation="relu")
        ]
        )

    def call(self, inputs, *args, **kwargs):
        return super().call(inputs, *args, **kwargs)

    # def call(self, input):
    #     return [ann(input) for ann in self.anns]
    

    # def call_vote(self, input):
    #     mean = tf.reduce_mean([ann(input) for ann in self.anns], axis=0)
    #     variance = tf.reduce_mean([tf.square(ann(input) - mean) for ann in self.anns], axis=0)
    #     return mean, variance


class TCN_enc_ANN_dec(keras.Model):
    """TCN encoder + ANN decoder model.
    Made to predict CPT from seismic data."""


    def __init__(self, tcn_encoder):

        self.tcn_encoder = tcn_encoder
        self.ann_decoder = ANN_committee()(self.tcn_encoder.output)

    def compile(self, **kwargs):
        self.ann_decoder.compile(**kwargs)

    def train_step(self, batch_data):
        X, y = batch_data

        latent_space = self.tcn_encoder(X)

        with tf.GradientTape() as tape:
            y_pred = self.ann_decoder(latent_space)
            losses = [self.ann_decoder.loss(y, pred) for pred in y_pred]
        

        for member, loss in zip(self.ann_decoder, losses):
            gradients = tape.gradient(loss, member.trainable_variables)
            self.tcn_encoder.optimizer.apply_gradients(zip(gradients, member.trainable_variables))

        committee_loss = self.ann_decoder.loss(y, self.ann_decoder.call_vote(latent_space)[0])
        committee_variance = self.ann_decoder.call_vote(latent_space)[1]

        return {'committee_loss': committee_loss, 'committee_variance': committee_variance}

    def predict(self, X):
        return self.ann_decoder.call_vote(self.tcn_encoder(X))

    def score(self, X, y):
        return self.ann_decoder.score(self.tcn_encoder(X), y)

    def __repr__(self):
        return 'TCN_enc_ANN_dec(tcn_encoder={}, ann_decoder={})'.format(self.tcn_encoder, self.ann_decoder)

    def __str__(self):
        return 'TCN_enc_ANN_dec(tcn_encoder={}, ann_decoder={})'.format(self.tcn_encoder, self.ann_decoder)
    

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
        keras.layers.LayerNormalization(),
        keras.layers.ZeroPadding2D(padding=((0, 0), (1, 1))), # 1, 1 padding because kernel is 3x3
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((1, 2)), # Reduce the depth of seismic to GM_len
        keras.layers.LayerNormalization(),
        keras.layers.Dropout(0.01)
    ], name='cnn_encoder')

    # Add more layers for shape reduction
    for _ in range((image_width-2*(3-1))//2):
        cnn_encoder.add(keras.layers.ZeroPadding2D(padding=((0, 0), (1, 1))))
        cnn_encoder.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        cnn_encoder.add(keras.layers.LayerNormalization())
        cnn_encoder.add(keras.layers.Dropout(0.01))


    cnn_encoder.add(keras.layers.Conv2D(latent_features, (1), activation='relu'))
    cnn_encoder.add(keras.layers.Reshape((-1, latent_features))) # Reshape to get features in the second dimension

    return cnn_encoder


def LSTM_decoder(latent_features=16):
    """LSTM decoder predicting CPT response from latent features."""
    lstm_decoder = keras.Sequential([
        keras.layers.InputLayer(input_shape=(None, latent_features)),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.LSTM(32, return_sequences=True),
        keras.layers.LSTM(16, return_sequences=True),
        keras.layers.Dense(3)
    ], name='lstm_decoder')

    return lstm_decoder


    

    
class Collapse_CNN(Model):
    """ Initializes a model with reducing dimensionality from a seismic image
    to a vector of latent features. The latent features are used to predict
    cpt response.
    """

    def __init__(self, latent_features, image_width, n_members=5):
        super(Collapse_CNN, self).__init__()
        self.cnn_encoder = CNN_pyramidal_encoder(latent_features=latent_features, image_width=image_width)
        self.ann_decoder = keras.models.Sequential([
             keras.layers.Conv1D(16, 1, activation='relu', padding='same'),
             keras.layers.Conv1D(32, 1, activation='relu', padding='same'),
             keras.layers.Conv1D(3, 1, activation='relu', padding='same')]
        )

    def call(self, X):
        latent_space = self.cnn_encoder(X)
        return self.ann_decoder(latent_space)
    
    def compile(self, **kwargs):
        super(Collapse_CNN, self).compile(**kwargs)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss = tf.keras.losses.MeanSquaredError()

    def fit(self, X, y, **kwargs):
        return super(Collapse_CNN, self).fit(X, y, **kwargs)

    def train_step(self, batch_data):
        X, y = batch_data

        latent_space = self.cnn_encoder(X)
        with tf.GradientTape() as tape:
            y_pred = self.ann_decoder(latent_space)
            loss = self.loss(y, y_pred)
        
        gradients = tape.gradient(loss, self.ann_decoder.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.ann_decoder.trainable_variables))

        return loss
    
    def predict(self, X):
        return self.ann_decoder.call_vote(self.cnn_encoder(X))
    
    

class Collapse_tree(Model):
    """
    Uses a CNN collapsing decoder and a tree based multi attribute regressor.
    """
    def __init__(self, Treedecoder):
        super(Collapse_tree, self).__init__()
        self.cnn_encoder = CNN_pyramidal_encoder(latent_features=16, image_width=11)
        self.tree_decoder = Treedecoder()

    def compile(self, **kwargs):
        super(Collapse_tree, self).compile(**kwargs)
        
    def train_step(self, batch_data):
        X, y = batch_data

        with tf.GradientTape() as tape:
            latent_space = self.cnn_encoder(X)
            self.tree_decoder.fit(latent_space, y)
            loss = 1 - self.tree_decoder.score(latent_space, y)

        gradients = tape.gradient(loss, self.cnn_encoder.trainable_variables)
        self.cnn_encoder.optimizer.apply_gradients(zip(gradients, self.cnn_encoder.trainable_variables))

        return loss


def ensemble_CNN_decoder(n_members=5):
    """1D CNN decoder with a committee of n_members."""

    print('More members are not implemented yet')
    ann_decoder = keras.models.Sequential([
        keras.layers.Conv1D(16, 5, activation='relu', padding='same'),
        keras.layers.Conv1D(32, 5, activation='relu', padding='same'),
        keras.layers.Conv1D(3, 5, activation='relu', padding='same')
    ], name='ann_decoder')

    return ann_decoder


def ensemble_CNN_model(n_members=5, latent_features=16, image_width=11, learning_rate=0.001):
    encoder = CNN_pyramidal_encoder(latent_features=latent_features, image_width=image_width)
    # decoder = ensemble_CNN_decoder(n_members=n_members)(encoder.output)
    decoder = LSTM_decoder(latent_features=latent_features)(encoder.output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model = Model(encoder.input, decoder)
    model.compile(loss='mae', optimizer=optimizer, metrics=['mse', 'mae'])
    return model, encoder


def predict_encoded_tree(encoder, tree, X): #, mask=None):
    """Predicts the target variable from encoded data using a tree based
    multi attribute regressor."""

    encoded = encoder(X).numpy()
    encoded = encoded.reshape(-1, encoded.shape[-1])
    pred = tree.predict(encoded)
    return pred.reshape(X.shape[0], -1, 3)