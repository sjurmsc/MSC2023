"""
Contains the model architectures so that they may easily be called upon.
"""
from distutils.command.sdist import sdist
import inspect
from sklearn.ensemble import RandomForestRegressor

# Some mess here that could be sorted out
from tensorflow import keras
from keras import backend as K, Model, Input, optimizers, layers
from keras.layers import Dense, Dropout, Conv1D, Conv2D, Layer, BatchNormalization, LayerNormalization
from keras.layers import Activation, SpatialDropout1D, SpatialDropout2D, Lambda, Flatten
from tensorflow_addons.layers import WeightNormalization
from numpy import array


class ResidualBlock(Layer):
    """
    If one would wish to write this as a class. Inspired by keras-tcn
    """
    def __init__(self,
                 dilation_rate: int,
                 nb_filters: int,
                 kernel_size: int,
                 padding: str,
                 activation: str = 'relu',
                 convolution_type: str = 'Conv2D',
                 dropout_rate: float = 0.,
                 kernel_initializer: str = 'he_normal',
                 use_batch_norm: bool = False,
                 use_layer_norm: bool = False,
                 use_weight_norm: bool = False,
                 **kwargs): # Any initializers for the Layer class
        """
        docstring here %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        """
        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.convolution_type = convolution_type # My addition
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm
        self.kernel_initializer = kernel_initializer
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
        self.res_output_shape = self.layers[-1].compute_output_shape(self.res_output_shape) # Not sure if compute output shape does anything here

    def build(self, input_shape):
        if self.convolution_type.lower() == 'conv1d': c_func = Conv1D
        else: c_func = Conv2D

        with K.name_scope(self.name):
            self.layers = []
            self.res_output_shape = input_shape

            for k in range(2):
                name = f'{self.convolution_type.lower()}_{k}'
                with K.name_scope(name):
                    conv = c_func(
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
                    if self.convolution_type.lower() == 'conv1d': d_func = SpatialDropout1D
                    else: d_func = SpatialDropout2D
                    self._build_layer(Activation(self.activation, name='Act_{}_{}'.format(self.convolution_type, k)))
                    self._build_layer(d_func(rate=self.dropout_rate, name='SDropout_{}'.format(k)))
    
            if self.nb_filters != input_shape[-1]:
                # 1x1 convolution mathes the shapes (channel dimension).
                name = 'matching_conv1D'
                with K.name_scope(name):

                    self.shape_match_conv = Conv1D(
                        filters=self.nb_filters,
                        kernel_size=1,
                        padding='same',
                        name=name,
                        kernel_initializer=self.kernel_initializer
                    )
            else:
                name = 'matching_identity'
                self.shape_match_conv = Lambda(lambda x: x, name=name)
            
            with K.name_scope(name):
                self.shape_match_conv.build(input_shape)
                self.res_output_shape = self.shape_match_conv.compute_output_shape(input_shape)
            
            self._build_layer(Activation(self.activation, name='Act_Conv_Blocks'))
            self.final_activation = Activation(self.activation, name='Act_Res_Block')
            self.final_activation.build(self.res_output_shape) # According to philipperemy this probably is not be necessary

            # Forcing keras to add layers in the list to self._layers
            for layer in self.layers:
                self.__setattr__(layer.name, layer)
            self.__setattr__(self.shape_match_conv.name, self.shape_match_conv)
            self.__setattr__(self.final_activation.name, self.final_activation)

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
                 kernel_size=3,
                 nb_stacks=1,
                 dilations=(1, 2, 4, 8, 16, 32),
                 padding='causal',
                 use_skip_connections=True,
                 dropout_rate=0.0,
                 return_sequences=False,
                 activation='relu',
                 convolution_type = 'Conv2D',
                 kernel_initializer='he_normal',
                 use_batch_norm=False,
                 use_layer_norm=False,
                 use_weight_norm=False,
                 **kwargs):
        
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.activation_name = activation
        self.convolution_type = convolution_type
        self.padding = padding
        self.kernel_initializer = kernel_initializer
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
            assert len(self.nb_filters) == len(self.dilations)
            if len(set(self.nb_filters)) > 1 and self.use_skip_connections:
                raise ValueError('Skip connections are not compatible'
                                 'with a list of filters, unless they are all equal.')
        if padding != 'causal' and padding != 'same':
            raise ValueError('Only \'causal\' or \'same\' padding are compatible for this layer.')
        
        # Initialize parent class (..which is Layer?)
        super(TCN, self).__init__(**kwargs)
    
    @property
    def receptive_field(self):
        return 1 + 2*(self.kernel_size-1)*self.nb_stacks*sum(self.dilations) # May need to pick the kernel dimension

    def build(self, input_shape):

        self.build_output_shape = input_shape

        self.residual_blocks = []
        total_num_blocks = self.nb_stacks * len(self.dilations)
        if not self.use_skip_connections:
            total_num_blocks += 1 # A cheap way to do a false case for below
    
        for s in range(self.nb_stacks):
            for i, d in enumerate(self.dilations):
                res_block_filters = self.nb_filters[i] if isinstance(self.nb_filters, list) else self.nb_filters
                self.residual_blocks.append(ResidualBlock(dilation_rate=d,
                                                          nb_filters=res_block_filters,
                                                          kernel_size=self.kernel_size,
                                                          padding=self.padding,
                                                          activation=self.activation_name,
                                                          convolution_type=self.convolution_type,
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

    # Not needed function that Philippe Remy wrote
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
        config['dropout_rate'] = self.dropout_rate
        config['return_sequences'] = self.return_sequences
        config['activation'] = self.activation_name
        config['convolution_type'] = self.convolution_type # May need another name
        config['use_batch_norm'] = self.use_batch_norm
        config['use_layer_norm'] = self.use_layer_norm
        config['use_weight_norm'] = self.use_weight_norm
        config['kernel_initializer'] = self.kernel_initializer
        return config


def TCN1D(trainX, param):
    """
    JR used trainx to decide dimensions of the architecture
    """
    pass
    

    # return model


def compiled_TCN(training_data, config):
    """
    @ Author: Sjur [in progress]
    Three temporal blocks as feature extractions

    Split into three for regression, and three for reconstruction

   This function only works for reconstruction at present moment 
    """
    nb_filters              = config['nb_filters']
    kernel_size             = config['kernel_size']
    dilations               = config['dilations']
    padding                 = config['padding']
    use_skip_connections    = config['use_skip_connections']
    dropout_rate            = config['dropout_rate']
    return_sequences        = config['return_sequences']
    activation              = config['activation']
    convolution_type        = config['convolution_type']
    lr                      = config['learn_rate']
    kernel_initializer      = config['kernel_initializer']
    use_batch_norm          = config['use_batch_norm']
    use_layer_norm          = config['use_layer_norm']
    use_weight_norm         = config['use_weight_norm']

    batch_size              = config['batch_size']
    epochs                  = config['epochs']
    conv_depth              = config['convolution_depth']

    # Data
    X, Y = training_data[1], training_data
    Y_reconstruct = array([dat.flatten() for dat in X])

    # input_shape = tuple([*X.shape[1:], nb_filters])
    input_layer = Input(shape=tuple(X.shape[1:]))

    # Feature Extraction module
    x = TCN(nb_filters=nb_filters,
            kernel_size=kernel_size,
            dilations=dilations,
            padding=padding,
            use_skip_connections=use_skip_connections,
            dropout_rate=dropout_rate,
            return_sequences=return_sequences,
            activation=activation,
            convolution_type=convolution_type,
            kernel_initializer=kernel_initializer,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_weight_norm=use_weight_norm
    )(input_layer)

    # Regression module
    for k in range(conv_depth):
        if k == 0: out = x
        else: out = r
        r = Conv1D(filters=nb_filters,
                    kernel_size=kernel_size,
                    padding = padding,
                    activation = 'relu',
                    name = 'Regression_{}'.format(k)
        )(out)
    r = Flatten()(r)
    r = Dense(Y[0].shape[1])(r)
    r = Activation('linear')(r)

    # Reconstruciton module
    conv_func = Conv1D
    dense_output_shape = X.shape[1]
    if convolution_type == 'Conv2D': conv_func = Conv2D; dense_output_shape = X.shape[1]*X.shape[2] # Not quite sure

    for k in range(conv_depth):
        x = conv_func(filters=nb_filters, 
                   kernel_size=kernel_size,
                   padding = padding,
                   activation='relu',
                   name = 'Reconstruction_{}'.format(k)
        )(x)
    x = Flatten()(x)
    x = Dense(dense_output_shape)(x)
    x = Activation('linear')(x)


    output_layer = [r, x] # Regression, reconstruction

    model = Model(inputs = input_layer, 
                  outputs = output_layer)
    model.compile(keras.optimizers.Adam(lr=lr, clipnorm=1.), loss='mean_squared_error')
    print(model.summary())
    model.fit(x=X, y=Y, batch_size=batch_size, epochs=epochs)
    
    return model


# Loss Function
import numpy as np
import tensorflow as tf

def model_loss(y_true, y_pred):
    loss = tf.reduce_mean(tf.norm((y_pred-y_true), ord=2))
    return loss

def weight_share_loss(y_true, y_pred):
    """
    As described in Mustafa et al. (2022)
    """
    yt_reg, yt_recon = y_true[0], y_true[1]
    yp_reg, yp_recon = y_pred[0], y_pred[1]
    recon_loss = model_loss(yt_recon, yp_recon)
    reg_loss = model_loss(yt_reg, yp_reg)
    total_loss = reg_loss + recon_loss
    return total_loss


