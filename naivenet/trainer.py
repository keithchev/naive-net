import os, glob, pdb
import numpy as np

import layers
import model
import losses
import optimizers

class Trainer(object):

    def __init__(self, rate_decay=1.0, batch_size=32, num_epochs=3, verbose=True):

        self._model     = None
        self._optimizer = None

        # training settings 
        self.rate_decay  = rate_decay
        self.batch_size  = batch_size
        self.num_epochs  = num_epochs
        self.verbose     = verbose

        self._data  = None
        self._loss  = None
        self._model = None

    # load the training and validation data
    @property
    def data(self):
        if self._data is None:
            print('No data loaded yet')
            return

        for k, d in self._data.iteritems():
            print('%s: %s' % (k, d.shape))

    @data.setter
    def data(self, _data):
        keys = set(['x_train', 'y_train', 'x_val', 'y_val'])

        if not type(_data)==dict or len(keys - set(_data.keys())) > 0:
            raise ValueError('data must have fields %s' % (list(keys)))

        self._data = _data
        self._N_train = self._data['x_train'].shape[0]
        self._N_val   = self._data['x_val'].shape[0]

        self._data['y_train'] = self._data['y_train'].reshape((self._N_train,))
        self._data['y_val']   = self._data['y_val'].reshape((self._N_val,))


    # set the model
    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, _model):
        if not isinstance(_model, model.Model):
            raise ValueError('model must be a Model instance')
        self._model = _model


    # set the optimizer 
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, _optimizer):
        if not isinstance(_optimizer, optimizers.Optimizer):
            raise ValueError('optimizer must be an Optimizer instance')
        self._optimizer = _optimizer
        self._init_optz_rate = self._optimizer.optz_rate


    # reset parameters that change during training
    # hard reset also resets the model itself (re-initz layer params and sets grads to zero)
    def reset(self, hard=False):
        '''
        reset the model and training parameters

        '''

        self.loss_at_batch     = []
        self.accuracy_at_epoch = {'train': [], 'val': []}

        # reset the optimizer - this resets the optz_rate (if rate decay was previously applied)
        self._optimizer.reset()

        # hard reset - re-initialize all layer parameters
        if hard: 
            self.model.reset()

            # set/reset the optimization dict associated with each parameter in each layer
            # This contains the per-parameter running averages used by momentumSGD, rmsprop, adam, etc
            for layer in self._model.layers:
                for parameter_name in layer.parameter_names:
                    self._optimizer.reset_parameter(layer[parameter_name])


    def train(self):
        '''
        train the model for the specific number of epochs

        '''
        
        if self._optimizer is None or self._model is None:
            raise ValueError('optimizer or model not set')

        batches_per_epoch = int(np.ceil(self._N_train / self.batch_size))

        for epoch_num in np.arange(self.num_epochs):
            for batch_num in np.arange(batches_per_epoch):

                loss = self._do_minibatch()
                self.loss_at_batch.append(loss)

            # decrement the optimization rate by the decay rate
            self._optimizer.optz_rate *= self.rate_decay

            self.accuracy_at_epoch['val'].append(self.accuracy('val'))
            self.accuracy_at_epoch['train'].append(self.accuracy('train'))

            if self.verbose:
                print('Epoch %d/%d | loss: %f | train acc: %f | val acc: %f' % (epoch_num, 
                                                                                self.num_epochs, 
                                                                                self.loss_at_batch[-1], 
                                                                                self.accuracy_at_epoch['train'][-1], 
                                                                                self.accuracy_at_epoch['val'][-1]))


    def _do_minibatch(self):

        batch_inds = np.random.choice(self._N_train, self.batch_size)

        x_batch = self._data['x_train'][batch_inds]
        y_batch = self._data['y_train'][batch_inds]

        loss = self.model.forward(x_batch, y_batch)
        self.model.backward()

        for layer in self.model.layers:
            for parameter_name in layer.parameter_names:
                self._optimizer.step(layer[parameter_name])

        return loss



    def accuracy(self, type_='train'):
        '''
        calculate accuracy on the validation data

        type_: 'train' or 'val'
        
        '''
        x = self._data['x_%s' % type_]
        y = self._data['y_%s' % type_]

        yp, out = self.model.predict(x)

        accuracy = 1.0*(y == yp).sum() / yp.shape[0]

        return accuracy



