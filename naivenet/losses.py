
import os, glob, pdb, datetime, time
import numpy as np



class Loss(object):

	def __init__(self):
		"""
		Generic loss function class

		"""

		# array of input activations 
		# dimension N x (feature dimensions C) where N is minibatch size

		self._x, self._y = None, None
		self._N, self._C = None, None

	# public methods
	def calc_loss(self, x, y):
		raise NotImplementedError('loss not implemented')

	def calc_dloss(self, x):
		raise NotImplementedError('dloss not implemented')
		

class SoftmaxLoss(Loss):
	"""
	Compute the softmax loss

	"""

	def __init__(self, num_classes=None):

		super(SoftmaxLoss, self).__init__()

		self.num_classes = num_classes


	def calc_loss(self, x, y):
		"""
		Forward pass 

		Inputs:
			x: N x C array of outputs from the last layer in the network
			y: vector of class labels of shape (N,) where y.max() < C
		"""

		if x.shape[1] != self.num_classes:
			raise ValueError('x.shape[1] is not equal to num_classes in SoftmaxLoss')
			return

		self._x = x
		self._N = x.shape[0]
		self._y = np.reshape(y, (self._N,)) # make sure y is 1D

		# softmax function (subtract max x for stability)
		softmax = np.exp(x - np.max(x, axis=1, keepdims=True))
		
		# normalize
		softmax /= np.sum(softmax, axis=1, keepdims=True)

		self._softmax = softmax
		
		# cross-entropy loss: sum log softmax over correct classes
		loss = -np.log(softmax[np.arange(self._N), self._y])

		# mean over minibatch
		loss = np.sum(loss) / self._N
		return loss


	def calc_dloss(self):
		""" 
		Gradient of the loss wrt the outputs in x

		Outputs:
			dloss_dx: N x C 
		"""

		# dloss wrt to output node activations in self._x
		# self._x is set during calc_loss above and is N x C 

		dloss_dx = self._softmax.copy()

		# loss depends linearly on activations corresponding to correct classes 
		dloss_dx[np.arange(self._N), self._y] -= 1
		
		# mean over mini-batch
		dloss_dx /= self._N

		return dloss_dx




