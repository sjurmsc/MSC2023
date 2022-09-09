"""
Contains the model architectures so that they may easily be called upon.
"""

from tensorflow import keras
from keras.layers import Input, Dense, Dropout, Conv1D, Conv2D



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


