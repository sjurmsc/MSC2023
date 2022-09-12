"""
Contains the model architectures so that they may easily be called upon.
"""
from sklearn.ensemble import RandomForestRegressor
from tensorflow import keras
from keras import backend as K
from keras.layers import Input, Dense, Dropout, Conv1D, Conv2D, Layer, BatchNormalization, LayerNormalization
from keras.layers import Activation, SpatialDropout1D, SpatialDropout2D, Lambda
from tensorflow_addons.layers import WeightNormalization


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
                 dropout_rate: float = 0., # Should be float?
                 kernel_initializer: str = 'he_normal', # ?????????????????????????
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
                

    def call():
        pass



def TemporalBlock2D(o, shape, filters, kernel_size, dilation_rate, dropout_rate):
    """
    Input, split into 1x1 convolution to maintain shape. The rest goes into
    dilated convolution layer

    Activation function: ReLU
    dropout layers at the ends, to prevent overfitting
    """

    i = Input(shape=shape)(o)

    # First Convolution
    p = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate, activation='relu')(i) # make sure weight norm is in there
    # Weight norm??
    p = Dropout(rate=dropout_rate)(p)

    # Second Convolution
    p = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate, activation='relu')(p)
    # Weight norm??
    p = Dropout(rate=dropout_rate)(p)


    # 1 x 1 Conv
    # conv1D(i)


    return o


def TemporalBlock1D(o, shape, filters, kernel_size, dilation_rate, dropout_rate):
    """
    """
    i = Input(shape=shape)(o) # This is wrong I think %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    # First Convolution
    p = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate, activation='relu')(i) # make sure weight norm is in there
    # Weight norm??
    p = Dropout(rate=dropout_rate)(p)

    # Second Convolution
    p = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate, activation='relu')(p)
    # Weight norm??
    p = Dropout(rate=dropout_rate)(p)


    return o


def TCN1D(trainX, param):
    """
    JR used trainx to decide dimensions of the architecture
    """
    pass
    

    # return model

def TCN2D(trainX, settings):
    """
    Three temporal blocks as feature extractions

    Split into three for regression, and three for reconstruction
    """
    filters = settings['filters']
    kernel_size = settings['kernel_size']
    dilation_rate = settings['dilation_rate']
    dropout_rate = settings['dropout_rate']

    shape = (trainX.shape[1], 1)

    # Feature Extraction module
    o = Input(shape)
    for filter, dilation in zip(filters, dilation_rate):
        o = TemporalBlock2D(o, shape, filter, kernel_size, dilation, dropout_rate)

    # Regression module

    # Reconstruciton module



# Loss Function
def weight_share_loss():
    pass


