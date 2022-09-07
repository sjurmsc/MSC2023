"""
Contains the model architectures so that they may easily be called upon.
"""

from tensorflow import keras
from keras.layers import Input, Dense, Dropout, Conv1D, Conv2D



def TemporalBlock2D(o, shape, filters, kernel_size, dilation_rate, dropout):
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
    p = Dropout(rate=dropout)(p)

    # Second Convolution
    p = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate, activation='relu')(p)
    # Weight norm??
    p = Dropout(rate=dropout)(p)


    # 1 x 1 Conv
    # conv1D(i)


    return o


def TemporalBlock1D(o, shape, filters, kernel_size, dilation_rate, dropout):
    """
    """
    i = Input(shape=shape)(o)
    
    # First Convolution
    p = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate, activation='relu')(i) # make sure weight norm is in there
    # Weight norm??
    p = Dropout(rate=dropout)(p)

    # Second Convolution
    p = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate, activation='relu')(p)
    # Weight norm??
    p = Dropout(rate=dropout)(p)

    return o


def TCN1D(trainX, param):
    """
    JR used trainx to decide dimensions of the architecture
    """
    pass
    

    # return model

def TCN2D(trainX, param):
    """
    Three temporal blocks as feature extractions

    Split into three for regression, and three for reconstruction
    """
    pass


def weight_share_cost():
    pass
