import os, sys
import numpy as np
import cPickle as pickle

def load_batches(batch_nums):

	root  = '.' + os.sep + 'data' + os.sep + 'cifar-10-batches-py'

	if type(batch_nums) != list: batch_nums = [batch_nums]

	x_cat, y_cat = None, None

	for batch_num in batch_nums:
		with open(root + os.sep + 'data_batch_%i' % batch_num, 'rb') as f:
			data = pickle.load(f)

		x = data['data']
		y = np.array(data['labels'])

		x = x.reshape(x.shape[0], 3, 32, 32).astype('float32')

		if x_cat is None: 
			x_cat, y_cat = x, y
			continue

		x_cat = np.concatenate((x_cat, x), axis=0)
		y_cat = np.concatenate((y_cat, y), axis=0)

	return x_cat, y_cat



