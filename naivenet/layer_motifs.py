import os, glob, pdb, datetime, time
import numpy as np

import layers
import model
import losses

def add_dense_bn(model_, D_in=None, D_out=None, momentum=0.9, atype='relu'):

    m.add_layer(layers.DenseLayer(D_in=input_dim, D_out=D_out))
    m.add_layer(layers.BatchNormLayer(D_in=D_out, momentum=momentum))
    m.add_layer(layers.ActivationLayer(atype=atype))

    return m

        


