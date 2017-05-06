import os, glob, pdb, datetime, time
import numpy as np

from naivenet.constants import _EPSILON
from naivenet.constants import _DTYPE
from naivenet.constants import _H

def test_model(model, num_sample=None):

	'''
	Compare analytic and numeric gradients for all parameters in a model
	num_sample is the number of gradient elements checked for each parameter (None -> all elements)

	'''

	# random data
	batch_size = 32

	x = np.random.randn(batch_size, *model.D_in)
	y = np.random.randint(model.loss.num_classes, size=batch_size)

	# re-initialize all of the parameters/gradients
	model.reset()

	# forward pass
	loss = model.forward(x, y)
	
	# backward pass to calc the analytic gradients in model.layers[:].param_grads
	model.backward()

	# function with which to calculate numeric gradients
	loss_function = lambda: model.forward(x, y)

	for layer_index, layer in enumerate(model.layers):

		# layers without params (activation layers) will be skipped here
		for param in layer.params:

			shape = layer.param_vals[param].shape
			if num_sample is not None:
				inds = []
				flat_inds = np.random.choice(np.prod(shape), num_sample)
				for flat_ind in flat_inds:
					inds.append(np.unravel_index(flat_ind, shape))
			else:
				inds = None

			grad  = calc_grad(loss_function, layer.param_vals[param], inds=inds)
			error = calc_error(grad, layer.param_grads[param], inds=inds)

			print('Layer %s (%s) %s error: %e' % (layer_index, layer.__class__.__name__, param, error))
			print('------------------------------------------------------------')

			if inds is not None:
				print('Numeric      Analytic')
				for i, ind in enumerate(inds):
					print('%0.6f     %0.6f' % (grad[i], layer.param_grads[param][ind]))

			print('\n')



def calc_error(numeric_grad, model_grad, inds=None):

	if inds is None:
		return np.max(np.abs(numeric_grad - model_grad) / (np.maximum(_EPSILON, np.abs(numeric_grad) + np.abs(model_grad))))

	else:
		model_grad_list = [model_grad[ind] for ind in inds]
		return np.max(np.abs(numeric_grad - model_grad_list) / (np.maximum(_EPSILON, np.abs(numeric_grad) + np.abs(model_grad_list))))


def calc_grad(loss_function, param_val, inds=None):

	if inds is None:
		param_grad = np.zeros_like(param_val)
		for ind in np.ndindex(*param_val.shape):
			param_grad[ind] = _calc_grad(loss_function, param_val, ind)
	else:
		param_grad = np.array([])
		for ind in inds:
			param_grad = np.append(param_grad, _calc_grad(loss_function, param_val, ind))
	
	return param_grad


def _calc_grad(loss_function, param_val, ind):

	param_val_ind = param_val[ind]
	
	param_val[ind] = param_val_ind + _H
	loss_plus = loss_function()

	param_val[ind] = param_val_ind - _H
	loss_minus = loss_function()

	# return param_val to the way it was
	param_val[ind] = param_val_ind

	param_grad_ind = (loss_plus - loss_minus) / (2*_H)

	return param_grad_ind






