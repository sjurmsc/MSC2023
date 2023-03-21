"""
Contains the model architectures so that they may be called by other scripts.
"""
from distutils.command.sdist import sdist
import inspect

# Some mess here that could be sorted out
from tensorflow import keras
from keras import backend as K, Model, Input, optimizers, layers
from keras.layers import Dense, Dropout, Conv1D, Conv2D, Layer, BatchNormalization, LayerNormalization
from keras.layers import Activation, SpatialDropout1D, SpatialDropout2D, Lambda, Flatten, LeakyReLU
from keras.layers import ZeroPadding1D, ZeroPadding2D
# from tensorflow_addons.layers import WeightNormalization
from numpy import array
from keras.utils.vis_utils import plot_model


class ResidualBlock(Layer):
    """
    If one would wish to write this as a class. Inspired by keras-tcn
    """
    def __init__(self,
                 dilation_rate: int,
                 nb_filters: int,
                 kernel_size,
                 padding: str,
                 activation: str = 'relu',
                 convolution_func: str = Conv2D,
                 dropout_type: str ='spatial',
                 dropout_rate: float = 0.,
                 kernel_initializer: str = 'he_normal',
                 use_batch_norm: bool = False,
                 use_layer_norm: bool = False,
                 use_weight_norm: bool = False,
                 **kwargs): # Any initializers for the Layer class
        """
        Creates a residual block for use in a TCN
        """
        # Checking whether dilations are a power of two
        assert (dilation_rate != 0) & ((dilation_rate & (dilation_rate - 1)) == 0), \
               'Dilations must be powers of 2'

        if convolution_func == Conv2D:
            self.dim = 2

            # Dilations only occur in depth; See Mustafa et al. 2021
            self.dilation_rate = (1, dilation_rate) # Height, width
        else:
            self.dim = 1
            self.dilation_rate = dilation_rate

        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer

        self.padding = padding
        self.activation = activation
        self.convolution_func = convolution_func # function for use in convolution layers
        self.dropout_type = dropout_type # Can be 'normal' or 'spatial'; decides what type of dropout layer is applied
        self.dropout_rate = dropout_rate

        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm

        # Variables to be filled
        self.layers = []
        self.shape_match_conv = None
        self.res_output_shape = None
        self.final_activation = None

        super(ResidualBlock, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        """
        Inspired by a function of the same name in another TCN implementation
        Not sure why input_shape is not used, but required as an input.
        """
        return [self.res_output_shape, self.res_output_shape]

    def _build_layer(self, layer):
        """Helper function for building layer
        Args:
            layer: Appends layer to internal layer list and builds it based on the current output
                   shape of ResidualBlocK. Updates current output shape.
        """
        self.layers.append(layer)
        self.layers[-1].build(self.res_output_shape)

        # This looks suspicious
        self.res_output_shape = self.layers[-1].compute_output_shape(self.res_output_shape) # Not sure if compute output shape does anything here

    def build(self, input_shape):

        with K.name_scope(self.name): # Gets the name from **kwargs
            
            self.layers = []
            self.res_output_shape = input_shape

            for k in range(2):
                name = f'{self.convolution_func.__name__}_{k}'
                with K.name_scope(name):

                    # Check out inputs here
                    conv = self.convolution_func(
                                                 filters=self.nb_filters,
                                                 kernel_size=self.kernel_size,
                                                 dilation_rate=self.dilation_rate,
                                                 padding=self.padding,
                                                 name=name,
                                                 kernel_initializer=self.kernel_initializer
                    )
                    if self.use_weight_norm:
                        from tensorflow_addons.layers import WeightNormalization

                        # WeightNormalization API is different than other Normalizations; requires wrapping
                        with K.name_scope('norm_{}'.format(k)):
                            conv = WeightNormalization(conv)
                    self._build_layer(conv)

                # Other Normalization types
                with K.name_scope('norm_{}'.format(k)):
                    if self.use_batch_norm:
                        self._build_layer(BatchNormalization())
                    elif self.use_layer_norm:
                        self._build_layer(LayerNormalization())
                    elif self.use_weight_norm:
                        pass # Already done above
                
                with K.name_scope('act_and_dropout_{}'.format(k)):
                    if self.dropout_type == 'normal':
                        d_func = Dropout
                        dname = 'Dropout'
                    if self.dropout_type == 'spatial':
                        dname = 'SDropout'
                        if self.dim == 1:
                            d_func = SpatialDropout1D
                        elif self.dim == 2:
                            d_func = SpatialDropout2D

                    self._build_layer(Activation(self.activation, name='Act_{}_{}'.format(self.convolution_func.__name__, k)))
                    self._build_layer(d_func(rate=self.dropout_rate, name='{}{}D_{}'.format(dname, self.dim, k)))
    
            if self.nb_filters != input_shape[-1]:
                # 1x1 convolution mathes the shapes (channel dimension).
                name = 'matching_conv'
                with K.name_scope(name):

                    self.shape_match_conv = self.convolution_func(
                        filters=self.nb_filters,
                        kernel_size=1,
                        padding='same',
                        name=name,
                        kernel_initializer=self.kernel_initializer # Why initialize this kernel with the same initializer?
                    )
            else:
                name = 'matching_identity'
                self.shape_match_conv = Lambda(lambda x: x, name=name)
            
            with K.name_scope(name):
                self.shape_match_conv.build(input_shape)
                self.res_output_shape = self.shape_match_conv.compute_output_shape(input_shape)
            
            # Names of these layers should be investigated
            self._build_layer(Activation(self.activation, name='Act_Conv_Blocks'))
            self.final_activation = Activation(self.activation, name='Act_Res_Block')
            self.final_activation.build(self.res_output_shape) # According to philipperemy this probably is not be necessary

            # Forcing keras to add layers in the list to self._layers
            for layer in self.layers:
                self.__setattr__(layer.name, layer)
            self.__setattr__(self.shape_match_conv.name, self.shape_match_conv)
            self.__setattr__(self.final_activation.name, self.final_activation) # I think this fixes the name issue

            super(ResidualBlock, self).build(input_shape) # This to make sure self.built is set to True

    def call(self, inputs, training=None, **kwargs):
        x1 = inputs
        for layer in self.layers:
            training_flag = 'traning' in dict(inspect.signature(layer.call).parameters)
            x1 = layer(x1, training=training) if training_flag else layer(x1)
        x2 = self.shape_match_conv(inputs)
        x1_x2 = self.final_activation(layers.add([x2, x1], name='Add_Res'))
        return [x1_x2, x1]


class TCN(Layer):
    """
    Creates a TCN layer.
    """

    def __init__(self,
                 nb_filters=64,
                 kernel_size=(3, 9),
                 nb_stacks=1,
                 dilations=(1, 2, 4, 8, 16, 32),
                 padding='causal',
                 use_skip_connections=True,
                 dropout_type = 'spatial',
                 dropout_rate=0.0,
                 return_sequences=False,
                 activation='relu',
                 convolution_func = Conv2D,
                 kernel_initializer='he_normal',
                 use_batch_norm=False,
                 use_layer_norm=False,
                 use_weight_norm=False,
                 reduce_rows=False,
                 **kwargs):
        
        self.return_sequences = return_sequences
        self.dropout_type = dropout_type
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.nb_filters = nb_filters
        self.activation_name = activation
        self.convolution_func = convolution_func
        self.padding = padding

        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm

        self.skip_connections = []
        self.residual_blocks = []
        self.layers_outputs = []
        self.build_output_shape = None
        self.slicer_layer = None  # in case return_sequence=False
        self.output_slice_index = None  # in case return_sequence=False
        self.padding_same_and_time_dim_unknown = False  # edge case if padding='same' and time_dim = None

        if self.use_batch_norm + self.use_layer_norm + self.use_weight_norm > 1:
            raise ValueError('Only one normalization can be specified at once.')
        
        if isinstance(self.nb_filters, list):
            assert len(self.nb_filters) == len(self.dilations) # Change of filter amount coincide with padding
            if len(set(self.nb_filters)) > 1 and self.use_skip_connections:
                raise ValueError('Skip connections are not compatible'
                                 'with a list of filters, unless they are all equal.')
        if padding != 'causal' and padding != 'same':
            raise ValueError('Only \'causal\' or \'same\' padding are compatible for this layer.')
        
        super(TCN, self).__init__(**kwargs)
    
    @property
    def receptive_field(self):
        return 1 + 2*(self.kernel_size-1)*self.nb_stacks*sum(self.dilations) # May need to pick the kernel dimension

    def build(self, input_shape):

        # Makes sure the i/o dims of each block are the same
        self.build_output_shape = input_shape

        self.residual_blocks = []
        total_num_blocks = self.nb_stacks * len(self.dilations)
        if not self.use_skip_connections:
            total_num_blocks += 1 # A cheap way to do a false case for below
    
        for s in range(self.nb_stacks):
            for i, d in enumerate(self.dilations):
                if isinstance(self.nb_filters, list):
                    res_block_filters = self.nb_filters[i] 
                else:
                    res_block_filters = self.nb_filters
                self.residual_blocks.append(ResidualBlock(dilation_rate=d,
                                                          nb_filters=res_block_filters,
                                                          kernel_size=self.kernel_size,
                                                          padding=self.padding,
                                                          activation=self.activation_name,
                                                          convolution_func=self.convolution_func,
                                                          dropout_type=self.dropout_type,
                                                          dropout_rate=self.dropout_rate,
                                                          use_batch_norm=self.use_batch_norm,
                                                          use_layer_norm=self.use_layer_norm,
                                                          use_weight_norm=self.use_weight_norm,
                                                          kernel_initializer=self.kernel_initializer,
                                                          name='residual_block_{}'.format(len(self.residual_blocks))))

        # this is done to force keras to add the layers in the list to self._layers
        for layer in self.residual_blocks:
            self.__setattr__(layer.name, layer)

        self.output_slice_index = None
        if self.padding == 'same':
            time = self.build_output_shape.as_list()[1]
            if time is not None:
                self.output_slice_index = int(self.build_output_shape.as_list()[1] / 2)
            else:
                # It will be known at call time
                self.padding_same_and_time_dim_unknown = True
        else:
            self.output_slice_index = -1 # causal case.
        self.slicer_layer = Lambda(lambda tt: tt[:, self.output_slice_index, :], name='Slice_Output')
        self.slicer_layer.build(self.build_output_shape.as_list())

    def compute_output_shape(self, input_shape):
        """
        Overridden in case keras uses it somewhere... no idea. Just trying to avoid future errors.
        """
        if not self.built:
            self.build(input_shape)
        if not self.return_sequences:
            batch_size = self.build_output_shape[0]
            batch_size = batch_size.value if hasattr(batch_size, 'value') else batch_size
            nb_filters = self.build_output_shape[-1]
            return [batch_size, nb_filters]
        else:
            # Compatibility tensorflow 1.x
            return [v.value if hasattr(v, 'value') else v for v in self.build_output_shape]

    def call(self, inputs, training=None, **kwargs):
        x = inputs
        self.layers_outputs = [x]
        self.skip_connections = []
        for res_block in self.residual_blocks:
            try:
                x, skip_out = res_block(x, training=training)
            except TypeError: # backwards compatibility
                x, skip_out = res_block(K.cast(x, 'float32'), training=training)
            self.skip_connections.append(skip_out)
            self.layers_outputs.append(x)
        
        if self.use_skip_connections:
            x = layers.add(self.skip_connections, name='Add_Skip_Connections')
            self.layers_outputs.append(x)
        
        if not self.return_sequences:
            # Case: time dimension is unknown. e.g. (bs, None, input_dim).
            if self.padding_same_and_time_dim_unknown:
                self.output_slice_index = K.shape(self.layers_outputs[-1])[1] // 2
            x = self.slicer_layer(x)
            self.layers_outputs.append(x)
        return x
    def get_config(self):
        """
        Returns the config of a the layer. This is used for saving and loading from a model
        :return: python dictionary with specs to rebuild layer
        """
        config = super(TCN, self).get_config() # Non-recursive, uses Layer.get_config(); key names must be standardized
        config['nb_filters'] = self.nb_filters
        config['kernel_size'] = self.kernel_size
        config['nb_stacks'] = self.nb_stacks
        config['dilations'] = self.dilations
        config['padding'] = self.padding
        config['use_skip_connections'] = self.use_skip_connections
        config['dropout_type'] = self.dropout_type
        config['dropout_rate'] = self.dropout_rate
        config['return_sequences'] = self.return_sequences
        config['activation'] = self.activation_name
        config['convolution_func'] = self.convolution_func
        config['use_batch_norm'] = self.use_batch_norm
        config['use_layer_norm'] = self.use_layer_norm
        config['use_weight_norm'] = self.use_weight_norm
        config['kernel_initializer'] = self.kernel_initializer
        return config


class CNN(Layer):

    def __init__(self,
                nb_filters=64,
                kernel_size=3,
                nb_stacks=3,
                padding='collapse',
                activation='relu',
                convolution_func = Conv2D,
                kernel_initializer='he_normal',
                dropout_rate = 0.001,
                use_dropout = False,
                use_batch_norm=False,
                use_layer_norm=False,
                use_weight_norm=False,
                **kwargs):
        
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size

        # If collapse padding is used, then the kernel size must be odd, and stacks must be sufficient to collapse the input
        self.nb_stacks = nb_stacks



        self.padding = padding
        self.activation = activation
        self.convolution_func = convolution_func
        self.pad_func = ZeroPadding2D
        self.dim = 2
        self.collapse = (padding=='collapse')
        if self.collapse:
            self.padding='valid'
            if kernel_size%2==0:
                raise ValueError('Kernel size must be odd for collapse padding.')
            if nb_stacks < int((kernel_size-1)/2):
                raise ValueError('Number of stacks must be sufficient to collapse the input.')


        # Compute the shape the data must have before being fed to the first layer
        self.data_shape = None

        self.kernel_initializer = kernel_initializer
        self.dropout_rate = dropout_rate
        self.use_dropout = use_dropout


        # Not sure if needed..
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm

        self.conv_blocks = []
        self.layers_outputs = []
        self.build_output_shape = None


        if convolution_func.__name__ == 'Conv1D':
            self.dim = 1
            self.pad_func = ZeroPadding1D
            if padding=='collapse':
                raise ValueError('Collapse padding is not supported for 1D convolutions.')


        super(CNN, self).__init__(**kwargs)

        
        

    def build(self, input_shape):

        self.build_output_shape = input_shape
        self.conv_blocks = []

        for k in range(self.nb_stacks):

            if self.collapse:
                #  the first dimension
                self.conv_blocks.append(self.pad_func(padding=((0, 0), ((self.kernel_size-1)/2,(self.kernel_size-1)/2)), name='pad_{}'.format(len(self.conv_blocks))))
                self.build_output_shape = self.conv_blocks[-1].compute_output_shape(self.build_output_shape)
                self.__setattr__(self.conv_blocks[-1].name, self.conv_blocks[-1])
            for i, f in enumerate([self.nb_filters]):
                conv_filters = self.nb_filters[i] if isinstance(self.nb_filters, list) else self.nb_filters
                self.conv_blocks.append(self.convolution_func(filters=conv_filters, 
                                                              kernel_size=self.kernel_size,
                                                              padding = self.padding,
                                                              activation=self.activation,
                                                              kernel_initializer=self.kernel_initializer,
                                                              name='convolution_layer_{}'.format(len(self.conv_blocks))))
        
        for layer in self.conv_blocks:
            self.__setattr__(layer.name, layer)


    def call(self, inputs, training=None, **kwargs):
        x = inputs
        self.layers_outputs = [x]
        for conv_block in self.conv_blocks:
            try:
                x = conv_block(x, training=training)
            except TypeError: # also backwards compatibiltiy
                x = conv_block(K.cast(x, 'float32'), training=training)
                self.layers_outputs.append(x)
        return x

    def get_config(self):
        config = super(CNN, self).get_config()
        config['nb_filters'] = self.nb_filters
        config['kernel_size'] = self.kernel_size
        config['nb_stacks'] = self.nb_stacks
        config['padding'] = self.padding
        config['activation'] = self.activation
        config['convolution_func'] = self.convolution_func
        config['kernel_initializer'] = self.kernel_initializer
        config['use_batch_norm'] = self.use_batch_norm
        config['use_layer_norm'] = self.use_layer_norm
        config['use_weight_norm'] = self.use_weight_norm
        return config


def compiled_TCN(training_data, config, **kwargs):
    """
    This function is to be called for initiating a model
    with provided configurations
    """
    nb_filters              = config['nb_filters']
    kernel_size             = config['kernel_size']
    nb_tcn_stacks           = config['nb_tcn_stacks']
    nb_reg_stacks           = config['nb_reg_stacks']
    nb_rec_stacks           = config['nb_rec_stacks']
    dilations               = config['dilations']
    padding                 = config['padding']
    use_skip_connections    = config['use_skip_connections']
    dropout_type            = config['dropout_type']
    dropout_rate            = config['dropout_rate']
    return_sequences        = config['return_sequences']
    activation              = config['activation']
    convolution_func        = config['convolution_func']
    learning_rate           = config['learn_rate']
    kernel_initializer      = config['kernel_initializer']
    use_batch_norm          = config['use_batch_norm']
    use_layer_norm          = config['use_layer_norm']
    use_weight_norm         = config['use_weight_norm']
    use_adversaries         = config['use_adversaries']

    batch_size              = config['batch_size']
    epochs                  = config['epochs']


    # Data
    X, y = training_data

    input_shape = tuple([*X.shape[1:], 1])
    input_layer = Input(shape=input_shape)

    # Feature Extraction module
    x = TCN(nb_filters=nb_filters,
            kernel_size=kernel_size,
            nb_stacks=nb_tcn_stacks,
            dilations=dilations,
            padding=padding,
            use_skip_connections=use_skip_connections,
            dropout_type = dropout_type,
            dropout_rate=dropout_rate,
            return_sequences=return_sequences,
            activation=activation,
            convolution_func=convolution_func,
            kernel_initializer=kernel_initializer,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_weight_norm=use_weight_norm,
            name='Feature_exctraction_module'
    )(input_layer)
    
    #print('receptive field is: {}'.format(x.receptive_field()))

    # Regression module
    # reg_ksize = y[0].shape[-1]/(nb_reg_stacks) + 1  # for 1d preserving the shape of the data
    # reg_ksize = int(reg_ksize)
    reg = CNN(nb_filters=nb_filters,
            kernel_size=kernel_size,
            nb_stacks=nb_reg_stacks,
            padding='same',
            activation=activation,
            convolution_func=convolution_func,
            kernel_initializer=kernel_initializer,
            name = 'Regression_module'
            )(x)   

    reg = convolution_func(1, kernel_size, padding=padding, activation='linear', name='regression_output')(reg)
    
    # Reconstruciton module
    rec = CNN(nb_filters=nb_filters,
            kernel_size=kernel_size,
            nb_stacks=nb_rec_stacks,
            padding=padding,
            activation=activation,
            convolution_func=convolution_func,
            kernel_initializer=kernel_initializer,
            name = 'Reconstruction_module'
            )(x)


    rec = convolution_func(1, kernel_size, padding=padding, activation='linear', name='reconstruction_output')(rec)

    output_layer = [reg, rec] # Regression, reconstruction

    if use_adversaries:
        seis_gen_model = Model(inputs=input_layer, outputs=rec)
        ai_gen_model   = Model(inputs=input_layer, outputs=reg)
        seis_disc_model = discriminator(output_layer[1].shape[1:], 3, name='seismic_discriminator')
        ai_disc_model = discriminator(output_layer[0].shape[1:], 3, name='ai_discriminator')


        model = multi_task_GAN([ai_disc_model, seis_disc_model],
                               [ai_gen_model, seis_gen_model], 
                               alpha=config['alpha'],
                               beta=config['beta'])

        generator_loss = keras.losses.MeanSquaredError()
        discriminator_loss = keras.losses.BinaryCrossentropy()

        generator_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.)
        discriminator_optimizer = keras.optimizers.Adam(learning_rate=learning_rate*0.1, clipnorm=1.) # Discriminators learn more slowly

        model.compile(g_optimizer=generator_optimizer, 
                      d_optimizer=discriminator_optimizer, 
                      g_loss=generator_loss, 
                      d_loss=discriminator_loss)
        # model.summary()
    else:
        model = Model(inputs = input_layer, 
                  outputs = output_layer)
        model.compile(keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.), loss={'regression_output' : 'mean_squared_error',
                                                                           'reconstruction_output' : 'mean_squared_error'})
        model.summary()

    History = model.fit(x=X, y=y, batch_size=batch_size, epochs=epochs, **kwargs)
    
    # Tree model here
    #model.ai_generator.



    return model, History


# Loss Functions
import numpy as np
import tensorflow as tf

def model_loss(y_true, y_pred):
    loss = tf.reduce_mean(tf.norm((y_pred-y_true), ord=2))
    return loss

def weight_share_loss(y_true, y_pred):
    """
    As described in Mustafa et al. (2021)
    """
    yt_reg, yt_recon = y_true[0], y_true[1]
    yp_reg, yp_recon = y_pred[0], y_pred[1]
    recon_loss = model_loss(yt_recon, yp_recon)
    reg_loss = model_loss(yt_reg, yp_reg)
    total_loss = reg_loss + recon_loss
    return total_loss

def discriminator(Input_shape, 
                  depth = 4, 
                  convolution_func=Conv1D, 
                  dropout = 0.1, 
                  name='discriminator'):
    """
    Descriminator model for use in adversarial learning
    """
    input_layer = Input(Input_shape)
    x = input_layer
    for _ in range(depth):
        x = convolution_func(1, kernel_size=4, padding='valid')(x)
        x = layers.BatchNormalization(scale=False)(x)
        x = Dropout(rate = dropout)(x)
        x = layers.LeakyReLU()(x)
    x = layers.Flatten()(x)
    output_score = Dense(1, activation='sigmoid')(x)
    return Model(input_layer, output_score, name=name)



from lightgbm import LGBMRegressor

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


# import sklearn random forest regressor
from sklearn.ensemble import RandomForestRegressor


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
    

def CNN_collapsing_encoder(latent_features, image_width, GM_dz=0.1):
    """2D CNN encoder collapsing the dimension first dimension down to 1.
    Made to predict features at centered trace from seismic data."""

    assert image_width % 2 == 1, 'width % 2 != 1'

    # input = keras.layers.Input(shape=(None, image_width, 1), ragged=True)

    image_shape = (image_width, None, 1)

    cnn_encoder = keras.Sequential([        
        keras.layers.InputLayer(input_shape=image_shape),
        keras.layers.ZeroPadding2D(padding=((0, 0), (1, 1))),
        keras.layers.Conv2D(16, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.ZeroPadding2D(padding=((0, 0), (1, 1))), # 1, 1 padding because kernel is 3x3
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((1, 2)), # Reduce the depth of seismic to GM_len
        keras.layers.BatchNormalization()
    ])

    # Add more layers for shape reduction
    for _ in range((image_width-2*(3-1))//2):
        cnn_encoder.add(keras.layers.ZeroPadding2D(padding=((0, 0), (1, 1))))
        cnn_encoder.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        cnn_encoder.add(keras.layers.BatchNormalization())


    cnn_encoder.add(keras.layers.Conv1D(latent_features, (1), activation='relu'))
    # cnn_encoder.add(keras.layers.Reshape((GM_len, latent_features))) # Reshape to get features in the second dimension



    return cnn_encoder



    
class Collapse_CNN(Model):
    """ Initializes a model with reducing dimensionality from a seismic image
    to a vector of latent features. The latent features are used to predict
    cpt response.
    """

    def __init__(self, latent_features, image_width, n_members=5):
        super(Collapse_CNN, self).__init__()
        self.cnn_encoder = CNN_collapsing_encoder(latent_features=latent_features, image_width=image_width)
        self.ann_decoder = keras.models.Sequential([
             keras.layers.Conv1D(16, 1, activation='relu', padding='same'),
             keras.layers.Conv1D(3, 1, activation='relu', padding='same')]
        )(self.cnn_encoder.output)

    def call(self, X):
        latent_space = self.cnn_encoder(X)
        return self.ann_decoder(latent_space)
    
    def train_step(self, batch_data):
        X, y = batch_data


        with tf.GradientTape() as tape:
            y_pred = self.cnn_encoder(X)
            loss = self.loss(y, y_pred)
        

        for member, l in zip(self.ann_decoder, loss):
            gradients = tape.gradient(l, member.trainable_variables)
            self.cnn_encoder.optimizer.apply_gradients(zip(gradients, member.trainable_variables))

        # committee_loss = self.ann_decoder.loss(y, self.ann_decoder.call_vote(latent_space)[0])
        # committee_variance = self.ann_decoder.call_vote(latent_space)[1]

        # return {'committee_loss': committee_loss, 'committee_variance': committee_variance}
        return loss
    
    def predict(self, X):
        return self.ann_decoder.call_vote(self.cnn_encoder(X))
    

class Collapse_tree(Model):
    """
    Uses a CNN collapsing decoder and a tree based multi attribute regressor.
    """
    def __init__(self, Treedecoder):
        super(Collapse_tree, self).__init__()
        self.cnn_encoder = CNN_collapsing_encoder(latent_features=16, image_width=11)
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
    ann_decoder = keras.models.Sequential([
        keras.layers.Conv1D(16, 1, activation='relu', padding='same'),
        keras.layers.Conv1D(32, 1, activation='relu', padding='same'),
        keras.layers.Conv1D(3, 1, activation='relu', padding='same')
    ])

    return ann_decoder


def ensemble_CNN_model(X, y, **kwargs):
    encoder = CNN_collapsing_encoder(latent_features=16, image_width=11)
    decoder = ensemble_CNN_decoder(n_members=5)(encoder.output)

    model = Model(encoder.input, decoder)
    model.compile(loss='mse', optimizer='adam')
    model.fit(X, y, **kwargs)
    return model