"""
Contains the model architectures so that they may easily be called upon.
"""
from sklearn.ensemble import RandomForestRegressor
from tensorflow import keras
from keras import backend as K
from keras.layers import Input, Dense, Dropout, Conv1D, Conv2D, Layer
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
                 dropout_rate: float = 0,
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

    def _build_layer(self, layer):
        """
        Assists in building layers
        """
        self.layers.append(layer)



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


