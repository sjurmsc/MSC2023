"""
Contains the model architectures so that they may easily be called upon.
"""

from tensorflow import keras
from keras.layers import Input, Dense, Dropout, Conv1D, Conv2D



def 2DTemporalBlock(o, shape, params):
    """
    Input, split into 1x1 convolution to maintain shape. The rest goes into
    dilated convolution layer

    Activation function: ReLU
    dropout layers at the ends, to prevent overfitting
    """
    o = Input(shape=shape)(o)



    # Output her nede
    return o


def 1DTemporalBlock():
    """
    """
    pass



def weight_share_cost():
