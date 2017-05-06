import os, glob, pdb
import numpy as np

import layers
import model
import losses
import constants

'''
Common first-order gradient descent methods. 

Inputs:
	- param_val: array of parameter values (i.e., a weight matrix)
	- param_grad: array of gradients
	- param_opt: dictionary of paremeter-specific running means needed by the optimizer

Outputs:
	- updated param_val (in-place)
	- updated param_opt
'''

class Optimizer(object):

	_EPSILON = constants.epsilon()

	def __init__(self, optz_rate):
		if optz_rate is None:
			raise ValueError('optimization rate must be specified')
		self._init_optz_rate = optz_rate
		self.reset()

	def initialize_opt(self, param_val):
		raise NotImplementedError('Optimzer initialization not implemented')

	def step(self, param_val, param_grad, param_opt):
		raise NotImplementedError('Optimizer step not implemented')

	def reset(self):
		'''
		reset common parameters to their original values
		'''
		self.optz_rate = self._init_optz_rate
		return self



class VanillaSGD(Optimizer):

	def __init__(self, optz_rate=None):
		Optimizer.__init__(self, optz_rate=optz_rate)

	def initialize_opt(self, param_val):
		return None

	def step(self, param_val, param_grad, param_opt=None):

		param_val -= self.optz_rate * param_grad
		return param_val, param_opt


class MomentumSGD(Optimizer):

	def __init__(self, optz_rate=None, momentum=0.9):
		Optimizer.__init__(self, optz_rate=optz_rate)
		self.momentum = momentum


	def initialize_opt(self, param_val):
		return {'velocity': np.zeros_like(param_val)}


	def step(self, param_val, param_grad, param_opt):

		velocity = param_opt.get('velocity')
		velocity = self.momentum * velocity - self.optz_rate * param_grad
		
		param_val += velocity
		param_opt['velocity'] = velocity

		return param_val, param_opt



class RMSProp(Optimizer):

	def __init__(self, optz_rate=None, beta2=0.99):
		Optimizer.__init__(self, optz_rate=optz_rate)

		self.decay_rate = decay_rate


	def initialize_opt(self, param_val):
		return {'running_moment2': np.zeros_like(param_val)}


	def step(self, param_val, param_grad, param_opt):

		running_moment2 = self.beta2 * param_opt['running_moment2'] + (1-self.beta2) * (param_grad**2)
		param_val -= self.optz_rate * param_grad / (np.sqrt(running_moment2) + self._EPSILON)

		param_opt['running_moment2'] = running_moment2

		return param_val, param_opt



class Adam(Optimizer):

	def __init__(self, optz_rate=None, beta1=0.9, beta2=0.999):
		Optimizer.__init__(self, optz_rate=optz_rate)

		self.beta1 = beta1
		self.beta2 = beta2
		

	def initialize_opt(self, param_val):

		return {'running_moment1': np.zeros_like(param_val),
				'running_moment2': np.zeros_like(param_val),
				'count': 0}


	def step(self, param_val, param_grad, param_opt):
		
		param_opt['count'] += 1

		running_moment1 = self.beta1 * param_opt['running_moment1'] + (1-self.beta1) * param_grad
		running_moment2 = self.beta2 * param_opt['running_moment2'] + (1-self.beta2) * (param_grad**2)

		rm1t = running_moment1 / (1 - self.beta1**param_opt['count'])
		rm2t = running_moment2 / (1 - self.beta2**param_opt['count'])

		param_val = self.optz_rate * rm1t / (np.sqrt(rm2t) + self._EPSILON)		

		param_opt['running_moment1'] = running_moment1
		param_opt['running_moment2'] = running_moment2

		return param_val, param_opt