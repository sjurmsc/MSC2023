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
    i = Input(shape=shape)(o)
    Conv2D()




    # 1 x 1 Conv
    # conv1D(i)



    # Output her nede
    return o


def 1DTemporalBlock():
    """
    """
    pass


def TCN1D(trainX, param):
    """
    JR used trainx to decide dimensions of the architecture
    """
    pass
    

    # return model

def TCN2D(trainX, param):
    pass


def weight_share_cost():
