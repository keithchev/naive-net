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

	def reset_parameter(self, parameter):
		raise NotImplementedError('Optimzer initialization not implemented')

	def step(self, parameter):
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

	def reset_parameter(self, parameter):
		pass

	def step(self, parameter):
		parameter.value -= self.optz_rate * parameter.gradient


class MomentumSGD(Optimizer):

	def __init__(self, optz_rate=None, momentum=0.9):
		Optimizer.__init__(self, optz_rate=optz_rate)
		self.momentum = momentum


	def reset_parameter(self, parameter):
		parameter.opt_data = {'velocity': np.zeros_like(parameter.value)}


	def step(self, parameter):

		velocity = parameter.opt_data.get('velocity')
		velocity = self.momentum * velocity - self.optz_rate * parameter.gradient
		
		parameter.value += velocity
		parameter.opt_data['velocity'] = velocity



class RMSProp(Optimizer):

	def __init__(self, optz_rate=None, beta2=0.99):
		Optimizer.__init__(self, optz_rate=optz_rate)
		self.decay_rate = decay_rate


	def reset_parameter(self, parameter):
		parameter.opt_data = {'running_moment2': np.zeros_like(parameter.value)}


	def step(self, parameter):

		running_moment2 = self.beta2 * parameter.opt_data['running_moment2'] + (1-self.beta2) * (parameter.gradient**2)
		parameter.value -= self.optz_rate * parameter.gradient / (np.sqrt(running_moment2) + self._EPSILON)

		parameter.opt_data['running_moment2'] = running_moment2



class Adam(Optimizer):

	def __init__(self, optz_rate=None, beta1=0.9, beta2=0.999):
		Optimizer.__init__(self, optz_rate=optz_rate)
		self.beta1 = beta1
		self.beta2 = beta2
		

	def reset_parameter(self, parameter):

		parameter.opt_data = {'running_moment1': np.zeros_like(parameter.value),
							'running_moment2': np.zeros_like(parameter.value),
							'count': 0}


	def step(self, parameter):
		
		parameter.opt_data['count'] += 1

		running_moment1 = self.beta1 * parameter.opt_data['running_moment1'] + (1-self.beta1) * parameter.gradient
		running_moment2 = self.beta2 * parameter.opt_data['running_moment2'] + (1-self.beta2) * (parameter.gradient**2)

		rm1t = running_moment1 / (1 - self.beta1**parameter.opt_data['count'])
		rm2t = running_moment2 / (1 - self.beta2**parameter.opt_data['count'])

		parameter.value = self.optz_rate * rm1t / (np.sqrt(rm2t) + self._EPSILON)		

		parameter.opt_data['running_moment1'] = running_moment1
		parameter.opt_data['running_moment2'] = running_moment2
