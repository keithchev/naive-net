
import numpy as np

'''
common node activation functions
Only the get() method is intended to be public (and is called by ActivationLayer)

Implements 'relu', 'sigmoid', and 'tanh'

'''


def get(activation_type):

    func_attr = '_%s'  % activation_type
    der_attr  = '_d%s' % activation_type

    if func_attr not in globals().keys():
        raise ValueError('Activation function %s not found' % activation_type)

    func = globals()[func_attr]
    der  = globals()[der_attr]

    return func, der


def _relu(x):
    out = x.copy()
    out[x < 0] = 0
    return out

def _drelu(x, dloss_dout):
    dloss_dx = dloss_dout.copy()
    dloss_dx[x < 0] = 0
    return dloss_dx


def _sigmoid(x):
    return 1.0/(1 + np.exp(-x))

def _dsigmoid(x, dloss_dout):
    sig = _sigmid(x)
    dloss_dx = dloss_dout * sig*(1-sig)
    return dloss_dx


def _tanh(x):
    return np.tanh(x)

def _dtanh(x, dloss_dout):
    tanh = _tanh(x)
    dloss_dx = dloss_dout * (1 - tanh*tanh)
    return dloss_dx

    



