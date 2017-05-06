import os, glob, pdb
import numpy as np

import layers
import losses
import model
import trainer
import optimizers
import constants

#from ..datasets import cifar10


def load_trainer(data, net):

	opt = optimizers.MomentumSGD(optz_rate=0.01)

	t = trainer.Trainer(rate_decay=1.0, batch_size=32, num_epochs=3, verbose=True)

	t.data      = data
	t.model     = net.reset()
	t.optimizer = opt.reset()

	t.reset()

	return t

def cnn2():

	filter_sz   = 5
	conv_pad    = 2
	conv_stride = 1
	pool_sz     = 2
	pool_stride = 2
	num_classes = 10

	m = model.Model(lambda_=1e-3)

	# weight initializer for ReLu units
	kernel_initializer  = lambda shape: np.sqrt(2.0/np.prod(shape[1:]))
	weights_initializer = lambda shape: np.sqrt(2.0/shape[0])


	# layer_defs.push({type:'conv', sx:5, filters:16, stride:1, pad:2, activation:'relu'});
	C, H, W = (3, 32, 32)
	num_filters = 16

	H_conv_out = 1 + (H + 2*conv_pad - filter_sz)/conv_stride
	W_conv_out = 1 + (W + 2*conv_pad - filter_sz)/conv_stride

	H_pool_out = 1 + (H_conv_out - pool_sz)/pool_stride
	W_pool_out = 1 + (W_conv_out - pool_sz)/pool_stride

	num_pool_vx = num_filters * H_pool_out * W_pool_out


	m.add_layer(layers.Conv2D(D_in=(C, H, W), 
							  D_out=(num_filters, H_conv_out, W_conv_out), 
							  filter_shape=(filter_sz, filter_sz), 
							  stride=conv_stride, 
							  pad=conv_pad, 
							  initializer=kernel_initializer))

	m.add_layer(layers.Activation('relu'))

	m.add_layer(layers.MaxPool2D(D_in=(num_filters, H_conv_out, W_conv_out), 
							     D_out=(num_filters, H_pool_out, W_pool_out), 
							     filter_shape=(pool_sz, pool_sz), 
							     stride=pool_stride))


	D_out = (num_filters, H_pool_out, W_pool_out)
	

	# layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
	C, H, W = D_out
	num_filters = 20

	H_conv_out = 1 + (H + 2*conv_pad - filter_sz)/conv_stride
	W_conv_out = 1 + (W + 2*conv_pad - filter_sz)/conv_stride

	H_pool_out = 1 + (H_conv_out - pool_sz)/pool_stride
	W_pool_out = 1 + (W_conv_out - pool_sz)/pool_stride

	num_pool_vx = num_filters * H_pool_out * W_pool_out

	m.add_layer(layers.Conv2D(D_in=(C, H, W), 
							  D_out=(num_filters, H_conv_out, W_conv_out), 
							  filter_shape=(filter_sz, filter_sz), 
							  stride=conv_stride, 
							  pad=conv_pad, 
							  initializer=kernel_initializer))

	m.add_layer(layers.Activation('relu'))

	m.add_layer(layers.MaxPool2D(D_in=(num_filters, H_conv_out, W_conv_out), 
							     D_out=(num_filters, H_pool_out, W_pool_out), 
							     filter_shape=(pool_sz, pool_sz), 
							     stride=pool_stride))

	D_out = (num_filters, H_pool_out, W_pool_out)


	# layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
	C, H, W = D_out
	num_filters = 20

	H_conv_out = 1 + (H + 2*conv_pad - filter_sz)/conv_stride
	W_conv_out = 1 + (W + 2*conv_pad - filter_sz)/conv_stride

	H_pool_out = 1 + (H_conv_out - pool_sz)/pool_stride
	W_pool_out = 1 + (W_conv_out - pool_sz)/pool_stride

	num_pool_vx = num_filters * H_pool_out * W_pool_out

	m.add_layer(layers.Conv2D(D_in=(C, H, W), 
							  D_out=(num_filters, H_conv_out, W_conv_out), 
							  filter_shape=(filter_sz, filter_sz), 
							  stride=conv_stride, 
							  pad=conv_pad, 
							  initializer=kernel_initializer))

	m.add_layer(layers.Activation('relu'))

	m.add_layer(layers.MaxPool2D(D_in=(num_filters, H_conv_out, W_conv_out), 
							     D_out=(num_filters, H_pool_out, W_pool_out), 
							     filter_shape=(pool_sz, pool_sz), 
							     stride=pool_stride))

	D_out = (num_filters, H_pool_out, W_pool_out)


	m.add_layer(layers.Flatten(D_in=D_out))

	# Final dense layer
	m.add_layer(layers.Dense(D_in=(np.prod(D_out),), 
							 D_out=(num_classes,), 
							 initializer=weights_initializer))

	# softmax loss
	m.loss_function = losses.SoftmaxLoss(num_classes=num_classes)

	return m


def cnn(input_shape=(3, 32, 32),
		conv_pad=2, conv_stride=1, 
		pool_stride=2, 
		num_classes=10, 
		filter_sz=5, pool_sz=2, 
		num_filters=10, 
		num_dense=10):

	'''
	generic CNN architecture with dimensions for cifar10

	'''

	C, H, W = input_shape

	H_conv_out = 1 + (H + 2*conv_pad - filter_sz)/conv_stride 	# 32
	W_conv_out = 1 + (W + 2*conv_pad - filter_sz)/conv_stride

	H_pool_out = 1 + (H_conv_out - pool_sz)/pool_stride			# 16
	W_pool_out = 1 + (W_conv_out - pool_sz)/pool_stride

	num_pool_vx = num_filters * H_pool_out * W_pool_out 		# 2560

	m = model.Model(lambda_=1e-3)

	# weight initializer for ReLu units
	kernel_initializer  = lambda shape: np.sqrt(2.0/np.prod(shape[1:]))
	weights_initializer = lambda shape: np.sqrt(2.0/shape[0])

	m.add_layer(layers.Conv2D(D_in=(C, H, W), 
							  D_out=(num_filters, H_conv_out, W_conv_out), 
							  filter_shape=(filter_sz, filter_sz), 
							  stride=conv_stride, 
							  pad=conv_pad, 
							  initializer=kernel_initializer))

	m.add_layer(layers.Activation('relu'))

	m.add_layer(layers.MaxPool2D(D_in=(num_filters, H_conv_out, W_conv_out), 
							     D_out=(num_filters, H_pool_out, W_pool_out), 
							     filter_shape=(pool_sz, pool_sz), 
							     stride=pool_stride))

	m.add_layer(layers.Flatten(D_in=(num_filters, H_pool_out, W_pool_out)))

	# (10, 16, 16)
	# First hidden layer (dense - batchnorm - relu)

	m.add_layer(layers.Dense(D_in=(num_pool_vx,), 
							 D_out=(num_dense,), 
							 initializer=weights_initializer))

	m.add_layer(layers.BatchNorm(D_in=(num_dense,)))
	m.add_layer(layers.Activation('relu'))

	# Final dense layer
	m.add_layer(layers.Dense(D_in=(num_dense,), 
							 D_out=(num_classes,), 
							 initializer=weights_initializer))

	# softmax loss
	m.loss_function = losses.SoftmaxLoss(num_classes=num_classes)

	return m


def net3():

	'''
	A generic dense three-layer net

	'''

	num_classes = 2
	D_in        = (2,)
	D_hidden    = (10,)
	D_out       = (num_classes,)
	bn_momentum = 0.9

	init_weight_std = .1

	m = model.Model(lambda_=1e-3)


	# First hidden layer (dense - batchnorm - relu)
	m.add_layer(layers.Dense(D_in=D_in, 
							 D_out=D_hidden, 
					    	 init_weight_std=init_weight_std))

	m.add_layer(layers.BatchNorm(D_in=D_hidden, momentum=bn_momentum))
	m.add_layer(layers.Activation('relu'))


	# Second hidden layer (dense - batchnorm - relu)
	m.add_layer(layers.Dense(D_in=D_hidden, 
					    	 D_out=D_hidden, 
					  		 init_weight_std=init_weight_std))

	m.add_layer(layers.BatchNorm(D_in=D_hidden, momentum=bn_momentum))
	m.add_layer(layers.Activation('relu'))


	# Final dense layer
	m.add_layer(layers.Dense(D_in=D_hidden, D_out=D_out, init_weight_std=init_weight_std))


	# softmax loss
	m.loss = losses.SoftmaxLoss(num_classes=num_classes)

	return m


def net1():

	'''
	a generic two-layer net
	'''

	num_classes = 2
	D_in   = (3,)
	D_out = (num_classes,)
	momentum    = 0.9

	m = model.Model(lambda_=1e-3)

	m.add_layer(layers.Dense(D_in=D_in, D_out=D_out))
	m.add_layer(layers.BatchNorm(D_in=D_hidden, momentum=momentum))
	m.add_layer(layers.Activation(atype='relu'))

	# model.add_layer(DropoutLayer(p=0.5))

	m.loss = losses.SoftmaxLoss(num_classes=num_classes)

	return m


def cifar_like(N=1000):

	x = np.random.randn(N, 3, 32, 32)
	y = np.random.randint(10, size=N)

	N_val = np.round(N*.15)

	x_val = np.random.randn(N_val, 3, 32, 32)
	y_val = np.random.randint(10, size=N_val)

	return {'x_train': x, 'y_train': y, 'x_val': x_val, 'y_val': y_val}



def random_data(N=1000):

	x = np.random.randn(N, 2)
	y = np.random.randint(2, size=N)

	return {'x_train': x, 'y_train': y, 'x_val': x, 'y_val': y}


def blob_data(N=1000, std=1.0):

	def _blobs(N, std):

		x_A = std * np.random.randn(N, 2) + [1, 1]
		x_B = std * np.random.randn(N, 2) + [-1, -1]

		x = np.concatenate((x_A, x_B), axis=0)
		y = np.concatenate((np.zeros((N, 1))+0, np.zeros((N, 1))+1))

		return x, y


	data = {}

	x, y = _blobs(N, std)
	data['x_train'] = x
	data['y_train'] = y.astype('int')

	x, y = _blobs(N, std)
	data['x_val'] = x
	data['y_val'] = y.astype('int')

	return data



