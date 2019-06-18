import keras
from keras.layers import Conv2D, BatchNormalization
from keras.advanced_actibations import LeakyReLU

def conv():

    layer = Conv2D(n_maps, kernel)
    layer = batchNormalization()(layer)
    layer = LeakyReLU()(layer)
