import os, pdb
import numpy as np

import activations
import constants
from parameter import Parameter

class Layer(object):

	_EPSILON = constants.epsilon()

	def __init__(self):
		"""
		Generic/abstract layer base class
		Initialize variables common to all layer types
		"""

		# array of input activations 
		# dimension N x (feature dimensions) where N is minibatch size
		self._x   = None

		# public variables
		#
		# placeholder for param dicts
		# 'params' here refers to all layer params that are trained

		self.D_in   = None
		self.D_out  = None
		self.parameter_names = []   # list of parameter keys: 'weights', 'bias', 'kernel', etc


	def __getitem__(self, key):
		return getattr(self, key)

	def reset(self):
		'''
		reset the layer's data 
		'''
		self._x = None

		for parameter_name in self.parameter_names:
			self[parameter_name].reset()

	def forward(self, x, mode='train'):
		raise NotImplementedError('Forward pass not implemented')

	def backward(self, dloss_dout):
		raise NotImplementedError('Backward pass not implemented')


class Dense(Layer):
	"""
	A dense/affine/fully-connected layer

	***Multidimensional inputs must be flattened before being passed to this layer***
	
	Inputs:
		D_in:  scalar; dimension of the input
		D_out: scalar; dimension of the output (number of nodes in this layer)
		initializer: function of kernel shape that returns std dev of initial weights
					 (to set constant std, use initializer=lambda _: std)
	"""

	def __init__(self, D_in=None, D_out=None, initializer=None, regularizer='L2'):

		super(Dense, self).__init__()

		if len(D_in) > 1 or len(D_out) > 1:
			raise ValueError('DenseLayer input/output must be one-dimensional')

		self.D_in  = D_in
		self.D_out = D_out

		if initializer is None:
			initializer = lambda shape: np.sqrt(1./shape[0])
		
		self.parameter_names = ['weights', 'bias']

		self.weights = Parameter(shape=(self.D_in[0], self.D_out[0]), 
								initializer=initializer,
								regularizer=regularizer,
								trainable=True,
								name='weights')

		self.bias   = Parameter(shape=self.D_out, 
								initializer='zeros',
								regularizer=None,
								trainable=True,
								name='bias')
		self.reset()


	def forward(self, x, mode='train'):
		"""
		Forward pass 

		Inputs:
			x: N x D_in array
		"""

		self._x = x
		self._N = x.shape[0]
		out = x.dot(self.weights.value) + self.bias.value
		return out


	def backward(self, dloss_dout):
		""" 
		Backward pass

		Inputs:
			dloss_dout: N x out_dim gradient of the output wrt the loss 
		"""

		dloss_db = dloss_dout.sum(axis=0)
		dloss_dw = (dloss_dout[:, None, :] * self._x[:, :, None]).sum(axis=0)
		dloss_dx = (dloss_dout.dot(self.weights.value.transpose())).reshape(self._N, *self.D_in)
		
		self.bias.gradient    = dloss_db
		self.weights.gradient = dloss_dw

		return dloss_dx


class Activation(Layer):

	def __init__(self, activation_type='relu'):

		super(Activation, self).__init__()
		self._func, self._der = activations.get(activation_type)

	def forward(self, x, mode='train'):
		self._x = x
		return self._func(self._x)

	def backward(self, dloss_dout):
		return self._der(self._x, dloss_dout)


class Flatten(Layer):

	def __init__(self, D_in=None):
		super(Flatten, self).__init__()
		self.D_in = D_in
		self.D_out = (np.prod(D_in),)

	def forward(self, x, mode='train'):
		self._N = x.shape[0]
		return x.reshape(self._N, -1)

	def backward(self, dloss_dout):
		return dloss_dout.reshape(self._N, *self.D_in)


class Dropout(Layer):

	def __init__(self, p=0.5):
		super(Dropout, self).__init__()
		self.p = p

	def forward(self, x, mode='train'):

		self._mode = mode
		if self._mode=='train':
			drop = np.random.rand(*x.shape) > self.p
			out = x/self.p
			out[drop] = 0
			self._drop = drop

		elif self._mode=='test':
			out = x

		return out

	def backward(self, dloss_dout):

		if self._mode=='train':
			dloss_dx = dloss_dout/self.p
			dloss_dx[self._drop] = 0

		elif self._mode=='test':
			dloss_dx = dloss_dout

		return dloss_dx


class BatchNorm(Layer):

	def __init__(self, D_in=None, momentum=0.9):

		'''
		D is the dimension of the input (i.e., number of nodes in the preceding layer)
		momentum determines timescale of running mean/var
		'''

		super(BatchNorm, self).__init__()

		if len(D_in) > 1:
			raise ValueError('batchNorm input must be one-dimensional')

		self.D_in     = D_in
		self.momentum = momentum

		self.parameter_names = ['gamma', 'beta']

		self.gamma = Parameter(shape=self.D_in, 
								initializer='ones',
								regularizer=None,
								trainable=True,
								name='gamma')

		self.beta = Parameter(shape=self.D_in, 
								initializer='zeros',
								regularizer=None,
								trainable=True,
								name='beta')

		self.reset()


	def reset(self):
		Layer.reset(self)

		self._running_mean = np.zeros(self.D_in)
		self._running_var  = np.zeros(self.D_in) 


	def forward(self, x, mode='train'):

		# note: x must be of shape NxD
		
		if mode == 'train':

			self._current_mean = x.mean(axis=0)
			self._current_var  = x.var(axis=0)

			self._running_mean = self.momentum * self._running_mean + (1 - self.momentum)*self._current_mean
			self._running_var  = self.momentum * self._running_var + (1 - self.momentum)*self._current_var

			xhat = (x - self._current_mean) / (self._current_var + self._EPSILON)**.5

		elif mode == 'test':
			xhat = (x - self._running_mean) / (self._running_var + self._EPSILON)**.5

		else:
			raise ValueError('Modes other than train and test not supported by BatchNorm')

		self._x    = x
		self._xhat = xhat
		out = self.gamma.value * xhat + self.beta.value

		return out


	def backward(self, dloss_dout):
		# mode is always train

		self.beta.gradient  = dloss_dout.sum(axis=0)
		self.gamma.gradient = (dloss_dout * self._xhat).sum(axis=0)

		# define intermediate variables for computing dloss_dx
		N     = self._x.shape[0]
		xcen  = self._x - self._current_mean
		scale = (self._current_var + self._EPSILON)**(-0.5)

		dloss_dxhat = self.gamma.value * dloss_dout

		dloss_dx = dloss_dxhat - dloss_dxhat.sum(axis=0)/N - scale**2 * xcen * (dloss_dxhat*xcen).sum(axis=0)/N
		dloss_dx *= scale

		return dloss_dx


class Conv2D(Layer):

	def __init__(self, D_in=None, D_out=None, filter_shape=None, stride=1, pad=0, initializer=None, regularizer='L2'):
		'''
		Two-dimensional convolutional layer

		Inputs:

		- D_in: input shape (depth, height, width)
		- D_out: output shape (depth, height, width)
		- filter_shape: kernel size (height, width)
		- self.stride: scalar stride (isotropic)
		- self.pad: zero-padding set during initialization and calculated to work with stride and filter shape
		- initializer: function of input dimension that returns std dev of initial weights
		  			 (to set constant std, use initializer=lambda _: std)
		
		'''

		super(Conv2D, self).__init__()

		self.D_in  = D_in 
		self.D_out = D_out 
		self.filter_shape = filter_shape

		if initializer is None: initializer = lambda shape: np.sqrt(1./np.prod(shape[1:]))
		
		self.initializer = initializer  

		self.pad    = pad
		self.stride = stride

		self.parameter_names = ['kernel', 'bias']

		self.kernel = Parameter(shape=(self.D_out[0], self.D_in[0], self.filter_shape[0], self.filter_shape[1]), 
								initializer=initializer,
								regularizer=regularizer,
								trainable=True,
								name='kernel')

		self.bias = Parameter(shape=self.D_out[0], 
								initializer='zeros',
								regularizer=None,
								trainable=True,
								name='bias')
		self.reset()


	def forward(self, x, mode='train'):
		'''
		Two-dimensional convolutional forward pass 

		Input:
		  x: Input image (N, F_in, H_in, W_in)

		Params:
		  'w': filter kernel (F_out, F_in, H_f, W_f)
		  'b': filter biases (F_out, )

		Output:
		  out: output image (N, F_out, H_out, W_out)

		'''

		pad, stride = self.pad, self.stride

		N, F_in, H_in, W_in   = x.shape
		F_out, F_in, H_f, W_f = self.kernel.shape

		H_out = 1 + (H_in + 2*self.pad - H_f) / self.stride
		W_out = 1 + (W_in + 2*self.pad - W_f) / self.stride

		if x.shape[1:] != self.D_in:
			raise ValueError('Input shape is %s but expected %s' % (x.shape[1:], self.D_in))

		if (F_out, H_out, W_out) != self.D_out:
			raise ValueError('Calculated output shape is %s but expected %s' % ((F_out, H_out, W_out), self.D_out))


		x_pad = np.pad(x, ((0,0), (0,0), (pad, pad), (pad, pad)), 'constant', constant_values=(0,))

		N, F_in, H_in, W_in = x_pad.shape

		H_off = int(np.floor((H_f - 1)/2))
		W_off = int(np.floor((W_f - 1)/2))

		H_list = np.arange(H_off, H_in - (H_f - H_off - 1), stride)
		W_list = np.arange(W_off, W_in - (W_f - W_off - 1), stride)

		out = np.zeros((N, F_out, H_out, W_out))

		# add biases to each output slice
		out += self.bias.value[None, :, None, None]

		# iterate over filter footprints (each separated by stride)
		for h_out, h in enumerate(H_list):
			for w_out, w in enumerate(W_list):

				x_crop = x_pad[:, :, (h - H_off):(h + H_f - H_off), (w - W_off):(w + W_f - W_off)]

				out[:, :, h_out, w_out] += (self.kernel.value[None, :, :, :, :] * x_crop[:, None, :, :, :]).sum(axis=(2,3,4))


		self._x = x
		self._x_pad = x_pad

		self._indexing_helpers = H_list, W_list, H_off, W_off

		return out


	def backward(self, dloss_dout):
		'''
		Two-dimensional convolutional backward pass 

		Input:
		  dloss_dout: gradient of the output volume (N, F_out, H_out, W_out)

		Params:
		  'w': filter kernel (F_out, F_in, H_f, W_f)
		  'b': filter biases (F_out, )

		Output:
		  dloss_dx: input image gradient (N, F_in, H_in, W_in)

		'''
		pad, stride = self.pad, self.stride

		F_out, F_in, H_f, W_f = self.kernel.shape
		H_list, W_list, H_off, W_off = self._indexing_helpers

		# gradient of the bias
		self.bias.gradient = dloss_dout.sum(axis=3).sum(axis=2).sum(axis=0)

		self.kernel.gradient = np.zeros_like(self.kernel.value)

		dloss_dx = np.zeros(self._x_pad.shape)

		for h_out, h in enumerate(H_list):
			for w_out, w in enumerate(W_list):
	
				x_crop = self._x_pad[:, :, (h - H_off):(h + H_f - H_off), (w - W_off):(w + W_f - W_off)]

				#  Fin, Fout, Hf, Wf                 	   N, Fout,            N, Fout, Fin, HH, WW             N, Fout, Fin, Hf, Wf
				self.kernel.gradient += np.sum(dloss_dout[:, :, h_out, w_out][:, :, None, None, None] * x_crop[:, None, :, :, :], axis=(0,))

				#        N, Fout, 																				  N, Fout,            N, Fout, Fin,                N,  Fout, Fin, hf, wf
				dloss_dx[:, :, (h - H_off):(h + H_f - H_off), (w - W_off):(w + W_f - W_off)] += np.sum(dloss_dout[:, :, h_out, w_out][:,:,None, None,None]*self.kernel.value[None, :, :, :, :], axis=(1,))


  		dloss_dx = dloss_dx[:, :, pad:-pad, pad:-pad]

		return dloss_dx


class MaxPool2D(Layer):

	def __init__(self, D_in=None, D_out=None, filter_shape=None, stride=1):
		'''
		Two-dimensional convolutional layer

		Inputs:

		- D_in: input shape (depth, height, width)
		- D_out: output shape (depth, height, width)
		- filter_shape: pool region shape (height, width)
		- self.stride: scalar stride (isotropic)
		
		'''

		super(MaxPool2D, self).__init__()

		self.D_in   = D_in 
		self.D_out  = D_out 
		self.stride = stride
		self.filter_shape = filter_shape


	def forward(self, x, mode='train'):
		'''
		Two-dimensional max pooling forward  

		Input:
		  x: Input image (N, F_in, H_in, W_in)

		Output:
		  out: output image (N, F_in, H_out, W_out)

		'''

		N, F, H, W = x.shape
		H_p, W_p = self.filter_shape

		H_list = np.arange(0, H - H_p + 1, self.stride)
		W_list = np.arange(0, W - W_p + 1, self.stride)

		out = np.zeros((N, F, len(H_list), len(W_list)))

		for h_out, h in enumerate(H_list):
			for w_out, w in enumerate(W_list):

				x_crop = x[:, :, h:h + H_p, w:w + W_p]
				out[:, :, h_out, w_out] = np.max(x_crop, axis=(2, 3))

		self._x = x
		self._indexing_helpers = H_list, W_list
		return out


	def backward(self, dloss_dout):
		'''
		Two-dimensional max pooling forward  

		Input:
		  dloss_dout: output gradients (N, F_in, H_out, W_out)

		Output:
		  dloss_dx: input gradients (N, F_in, H_in, W_in)

		'''

		N, F, H, W = self._x.shape
		H_p, W_p = self.filter_shape
		H_list, W_list = self._indexing_helpers

		dloss_dx = np.zeros_like(self._x)
		mask     = np.zeros((N*F, H_p*W_p))

		for h_out, h in enumerate(H_list):
			for w_out, w in enumerate(W_list):

				x_crop = self._x[:, :, h:h + H_p, w:w + W_p].reshape(N*F, -1)

				inds = np.argmax(x_crop, axis=1)

				mask *= 0
				mask[np.arange(N*F), inds] = dloss_dout[:, :, h_out, w_out].reshape(N*F)

				dloss_dx[:, :, h:h + H_p, w:w + W_p] = mask.reshape(N, F, H_p, W_p)

		return dloss_dx
