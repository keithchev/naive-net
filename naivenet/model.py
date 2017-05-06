import os, glob, pdb, datetime, time
import numpy as np

import losses
import layers

class Model(object):

	'''
	Construct a neural network 
	Public methods:
		- add_layer(<layer instance>)
		- loss(<loss instance>)
		- regz(<scalar>)
		- forward(x, y)
		- backward()
		- predict(x)
	'''

	def __init__(self, regz=0.0):

		# private variables
		self._loss = None	# loss function - must be set externally after initialization
		self._regz = regz 	# L2 regularization constant
		self._num_layers = 0

		self._x, self._y = None, None

		self.layers = []	# list of layers (Layer class instances)


	def add_layer(self, layer):

		if not isinstance(layer, layers.Layer):
			raise ValueError('layer must be a Layer instance')

		if len(self.layers)==0:
			self.D_in = layer.D_in

		self.layers.append(layer)


	@property
	def loss(self):
		return self._loss

	@loss.setter
	def loss(self, _loss):
		if not isinstance(_loss, losses.Loss):
			raise ValueError('Loss function must be a Loss instance')
		self._loss = _loss


	def reset(self):
		for layer in self.layers:
			layer.reset()
		return self


	def forward(self, x, y):
		'''
		Do a forward pass given x, which is N x (samples dims), and targets/labels in y
		
		Inputs:
			x: N x (dimensions of each sample) array
			y: N x 1 list of labels

		Returns:
			loss: a scalar

		'''

		self._x, self._y = x, y

		if self._loss is None:
			raise ValueError('Loss function not set')
			return 

		self._num_layers = len(self.layers)

		out = self.layers[0].forward(x)

		for i in range(1, self._num_layers):
			out = self.layers[i].forward(out)

		loss = self.loss.calc_loss(out, y)

		if self._regz > 0:
			for layer in self.layers:
				if 'w' in layer.params:
					loss += 0.5*self._regz * (layer.param_vals['w']**2).sum()

		return loss


	def backward(self):
		''' Do a backward pass given x and y stored from the forward pass

		Returns: Nothing; only updates gradients in each layer.param_grads in self.layers
		'''

		if self._x is None:
			raise ValueError('Must call model.forward before model.backward')

		dloss_dout = self.loss.calc_dloss()

		for i in np.arange(self._num_layers-1, -1, -1):
			dloss_dout = self.layers[i].backward(dloss_dout)

		if self._regz > 0:
			for layer in self.layers:
				if 'w' in layer.params:
					layer.param_grads['w'] += self._regz * layer.param_vals['w']


	def predict(self, x):
		'''
		calculate predictions for y given x

		These are simply the argmax of the output of the last layer
		'''

		mode = 'test'

		out = self.layers[0].forward(x, mode=mode)

		for i in range(1, self._num_layers):
			out = self.layers[i].forward(out, mode=mode)

		return np.argmax(out, axis=1), out




	def info(self):

		'''
		print readable summary of layers/dimensions

		'''

		print('***Model info***')
		print('Regularization: %f' % self._regz)
		print('Layers:')

		for layer in self.layers:
			print('%s    %s --> %s' % (layer.__class__.__name__, layer.D_in, layer.D_out))

		print('%s (%s classes)' % (self._loss.__class__.__name__, self._loss.num_classes))

