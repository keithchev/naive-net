import numpy as np

class Parameter(object):

    '''
    A Parameter

    Methods:
        .reset
        .value
        .gradient
        .optimization_data

    '''

    def __init__(self, name, shape,
                        initializer='zeros',
                        regularizer=None,
                        trainable=True,
                        dtype='float32'):


        if initializer is None: initializer = 'zeros'

        self.name  = name
        self.shape = shape
        self.dtype = dtype
        self.initializer = initializer
        self.regularizer = regularizer
        self.trainable   = trainable

        self.optimization_data = {}


    def reset(self):

        self.optimization_data = {}     
        self.gradient = np.zeros(self.shape, dtype=self.dtype)

        if self.initializer=='zeros':
            self.value = np.zeros(self.shape, dtype=self.dtype)

        if self.initializer=='ones':
            self.value = np.ones(self.shape, dtype=self.dtype)

        if callable(self.initializer):
            self.value = np.random.randn(*self.shape) * self.initializer(self.shape)


    def regularize(self, lambda_, mode='loss'):

        loss = 0

        if self.regularizer is None: return loss

        if self.regularizer=='L2':

            if mode=='loss':
                loss = 0.5*lambda_*(self.value**2).sum()

            elif mode=='gradient':
                self.gradient += lambda_*self.value

            else:
                raise ValueError('Invalid regularization mode')

        return loss