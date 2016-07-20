import re
import gc
import time
from iterable_queue import IterableQueue
from multiprocessing import Process
from subprocess import check_output
from categorical import Categorical
from token_map import UNK
from unigram_dictionary import UnigramDictionary
import numpy as np
from theano import shared
import os
import sys


class TheanoMinibatcher(object):
	'''
	This generates a theano shared variable storing the full dataset
	-- all training examples.  When the theano device setting is the
	GPU, shared variables are stored on the GPU, so this has the
	effect of loading the full dataset onto the GPU.

	One of the return values is a (set of) symbolic theano variable(s)
	corresponding to a single minibatch of the data.  This symbolic
	variable can be used to set up the training function.  What will
	happen during training is that this variable acts as a sliding
	"window" on the full dataset, selecting each minibatch in turn,
	even though the entire dataset is loaded into GPU memory.

	The indexing that causes the symbolic minibatch to address different
	parts of the dataset is itself a shared variable, and it can be
	updated using an update tuple provided to the updates list of a
	theanod function.  The necessary update tuple is also provided as
	a return value, so that it can be incorporated into the training
	function
	'''

	def __init__(self, batch_size=1000, dtype="float32", num_dims=2):
		self.batch_size = batch_size
		self.dtype = dtype
		self.num_dims = num_dims

		self._setup_batching()


	def _initialize_data_container(self, num_dims, dtype):

		# Validate num_dims
		if(num_dims < 1 or not isinstance(num_dims, int)):
			raise ValueError(
				'TheanoMinibatcher: num_dims must be an integer equal to or '
				'greater than 1.'
			)

		# Create the first dim, which houses the dataset
		data_container = []
		num_dims -= 1

		# Repeatedly add a nested dimension, so that we have num_dims nestings
		nested_container_handle = data_container
		for dim_num in range(num_dims):
			new_inner_container = []
			nested_container_handle.append(new_inner_container)
			nested_container_handle = new_inner_container

		return np.array(data_container, dtype=dtype)

	def reset(self):
		'''
		Reset the internal batch_num pointer to the start of the dataset
		'''
		self.batch_num.set_value(0)


	def _setup_batching(self):

		# Make an empty shared variable that will store the dataset
		# Although empty, we can setup the relationship between the
		# minibatch variable and the full dataset
		self.dataset = shared(
			self._initialize_data_container(self.num_dims, self.dtype)
		)

		# Make minibatch by indexing into the dataset
		self.batch_num = shared(np.int32(0))
		batch_start = self.batch_num * self.batch_size
		batch_end = batch_start + self.batch_size
		self.batch = self.dataset[batch_start : batch_end,]

		# Define an update that moves the batch window through the dataset
		self.updates = [(self.batch_num, self.batch_num+1)]


	def load_dataset(self, dataset):
		# Load the dataset onto the gpu
		self.dataset.set_value(dataset)
		# Determine the total number of minibatches
		self.num_batches = int(np.ceil(len(dataset) / float(self.batch_size)))
		return self.num_batches


	def get_batch(self):
		return self.batch


	def get_updates(self):
		return self.updates


	def get_num_batches(self):
		return self.num_batches
