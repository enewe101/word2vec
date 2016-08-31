from noise_contrast import get_noise_contrastive_loss
import os
import numpy as np
from dataset_reader import DatasetReader, default_parse
from theano_minibatcher import (
	TheanoMinibatcher, NoiseContrastiveTheanoMinibatcher
)

import os
# Only import theano and lasagne if environment permits it
exclude_theano_set = 'EXCLUDE_THEANO' in os.environ
if exclude_theano_set and int(os.environ['EXCLUDE_THEANO']) == 1:
	# Don't import theano and lasagne
	pass
else:
	from theano import tensor as T, function, shared
	import lasagne
	from lasagne.layers import (
		get_output, InputLayer, EmbeddingLayer, get_all_params,
		get_all_param_values
	)
	from lasagne.init import Normal
	from lasagne.updates import nesterov_momentum


def row_dot(matrixA, matrixB):
	C = (matrixA * matrixB).sum(axis=1)
	return C


def sigmoid(tensor_var):
	return 1/(1+T.exp(-tensor_var))


def word2vec(

		# Input / output options
		files=[],
		directories=[],
		skip=[],
		save_dir=None,
		read_data_async=True,
		num_processes=3,
		max_queue_size=0,
		parse=default_parse,

		# Batching options
		num_epochs=5,
		batch_size = 1000,  # Number of *signal* examples per batch
		macrobatch_size = 100000,

		# Dictionary options
		unigram_dictionary=None,
		load_dictionary_dir=None,
		min_frequency=10,

		# Sampling options
		noise_ratio=15,
		kernel=[1,2,3,4,5,5,4,3,2,1],
		t = 1.0e-5,

		# Embedding options
		num_embedding_dimensions=500,
		query_embedding_init=None,
		context_embedding_init=None,

		# Learning rate options
		learning_rate=0.1,
		momentum=0.9,

		# Verbosity option
		verbose=True
	):

	'''
	Helper function that handles all concerns involved in training
	A word2vec model using the approach of Mikolov et al.  It surfaces
	all of the options.

	For customizations going beyond simply tweeking existing options and
	hyperparameters, substitute this function by writing your own training
	routine using the provided classes.  This function would be a starting
	point for you.
	'''

	# Make a Word2VecMinibatcher, pass through parameters sent by caller
	reader = DatasetReader(
		files=files,
		directories=directories,
		skip=skip,
		macrobatch_size=macrobatch_size,
		max_queue_size=max_queue_size,
		noise_ratio=noise_ratio,
		num_processes=num_processes,
		unigram_dictionary=unigram_dictionary,
		load_dictionary_dir=load_dictionary_dir,
		min_frequency=min_frequency,
		t=t,
		kernel=kernel,
		parse=parse,
		verbose=verbose
	)

	# Prepare the dataset reader (this produces the counter_sampler stats)
	if not reader.is_prepared():
		if verbose:
			print 'preparing dictionaries...'
		reader.prepare(save_dir=save_dir)

	# Make a symbolic minibatcher
	minibatcher = NoiseContrastiveTheanoMinibatcher(
		batch_size=batch_size,
		noise_ratio=noise_ratio,
		dtype="int32",
		num_dims=2
	)

	# Make a Word2VecEmbedder object, feed it the combined input.
	# Note that the full batch includes noise examples and signal_examples
	# so is larger than batch_size, which is the number of signal_examples
	# only per batch.
	full_batch_size = batch_size * (1 + noise_ratio)
	embedder = Word2VecEmbedder(
		input_var=minibatcher.get_batch(),
		batch_size=full_batch_size,
		vocabulary_size=reader.get_vocab_size(),
		num_embedding_dimensions=num_embedding_dimensions,
		query_embedding_init=query_embedding_init,
		context_embedding_init=context_embedding_init
	)

	# Architectue is ready.  Make the loss function, and use it to create 
	# the parameter updates responsible for learning
	loss = get_noise_contrastive_loss(embedder.get_output(), batch_size)
	updates = nesterov_momentum(
		loss, embedder.get_params(), learning_rate, momentum
	)

	# Include minibatcher updates, which cause the symbolic batch to move
	# through the dataset like a sliding window
	updates.update(minibatcher.get_updates())

	# Use the loss function and the updates to compile a training function.
	# Note that it takes no inputs because the dataset is fully loaded using
	# theano shared variables
	train = function([], loss, updates=updates)

	# Iterate through the dataset, training the embeddings
	for epoch in range(num_epochs):

		if verbose:
			print 'starting epoch %d' % epoch

		if read_data_async:
			macrobatches = reader.generate_dataset_parallel()
		else:
			macrobatches = reader.generate_dataset_serial()

		macrobatch_num = 0
		for signal_macrobatch, noise_macrobatch in macrobatches:

			macrobatch_num += 1
			if verbose:
				print 'running macrobatch %d' % (macrobatch_num - 1)

			minibatcher.load_dataset(signal_macrobatch, noise_macrobatch)
			losses = []
			for batch_num in range(minibatcher.get_num_batches()):
				if verbose:
					print 'running minibatch', batch_num
				losses.append(train())
			if verbose:
				print '\taverage loss: %f' % np.mean(losses)

	# Save the model (the embeddings) if save_dir was provided
	if save_dir is not None:
		embedder.save(save_dir)
		reader.save_dictionary(save_dir)

	# Return the trained embedder and the dictionary mapping tokens
	# to ids
	return embedder, reader



class Word2VecEmbedder(object):

	def __init__(
		self,
		input_var,
		batch_size,
		vocabulary_size=100000,
		query_vocabulary_size=None,
		context_vocabulary_size=None,
		num_embedding_dimensions=500,
		query_embedding_init=None,
		context_embedding_init=None
	):

		self.query_embedding_init = query_embedding_init
		if query_embedding_init is None:
			self.query_embedding_init = Normal()

		self.context_embedding_init = context_embedding_init
		if context_embedding_init is None:
			self.context_embedding_init = Normal()

		# If only vocabular_size is specified, then both 
		# query_vocabulary_size and context_vocabulary_size take on that 
		# value, but ppecific values given for query_ or 
		# context_vocabulary_size override.
		self.query_vocabulary_size = vocabulary_size
		if query_vocabulary_size is not None:
			self.query_vocabulary_size = query_vocabulary_size

		self.context_vocabulary_size = vocabulary_size
		if context_vocabulary_size is not None:
			self.context_vocabulary_size = context_vocabulary_size

		self.input_var = input_var
		self.batch_size = batch_size
		self.num_embedding_dimensions = num_embedding_dimensions
		self._embed = None

		# Every row (example) in a batch consists of a query word and a
		# context word.  We need to learn separate embeddings for each,
		# and so, the input will be separated into all queries and all
		# contexts, and handled separately.
		self.query_input = input_var[:,0]
		self.context_input = input_var[:,1]

		# Make separate input layers for query and context words
		self.l_in_query = lasagne.layers.InputLayer(
			shape=(self.batch_size,), input_var=self.query_input
		)
		self.l_in_context = lasagne.layers.InputLayer(
			shape=(self.batch_size,), input_var=self.context_input
		)

		# Make separate embedding layers for query and context words
		self.l_embed_query = lasagne.layers.EmbeddingLayer(
			incoming=self.l_in_query,
			input_size=self.query_vocabulary_size,
			output_size=self.num_embedding_dimensions,
			W=self.query_embedding_init
		)
		self.l_embed_context = lasagne.layers.EmbeddingLayer(
			incoming=self.l_in_context,
			input_size=self.context_vocabulary_size,
			output_size=self.num_embedding_dimensions,
			W=self.context_embedding_init
		)

		# Obtain the embedded query words and context words
		self.query_embedding = get_output(self.l_embed_query)
		self.context_embedding = get_output(self.l_embed_context)

		# We now combine the query and context embeddings, taking the
		# dot product between corresponding queries and contexts.
		# We can express this as an element-wise multiplication between
		# the array of all query embeddings with all context embeddings
		# Summing along the rows of the resulting matrix yields the dot
		# product for each query-context pair.
		self.match_scores = row_dot(
			self.query_embedding,
			self.context_embedding
		)
		#self.match_scores = T.dot(
		#	self.query_embedding, self.context_embedding.T
		#).diagonal()

		# Finally apply the sigmoid activation function
		self.output = sigmoid(self.match_scores)


	def embed(self, token_id):
		'''
		Return the vectorspace embedding for the token.

		INPUTS
		* token [int] or [iterable : int]: Integer representation of a
			single token (e.g. a word), or a list of tokens.
		'''
		if self._embed is None:
			self._compile_embed()

		# This function is overloaded to accept a single token_id or
		# a list thereof.  Resolve that here.
		if isinstance(token_id, int):
			token_ids = [token_id]
		else:
			token_ids = token_id

		return self._embed(token_ids)


	def _compile_embed(self):
		'''
		Compiles the embedding function.  This is a separate theano
		function from that used during training of the network, but it
		produces exactly the same embeddings.  After being compiled
		once (usually the first time self.embed() is called), it will
		remain up-to-date, supplying the correct embeddings even if
		more training occurs.  This is because it references the same
		theano shared variables for its embedding parameters as the
		embedding layer used in training.

		(No Inputs)
		OUTPUTS
		* [None]
		'''
		input_tokens = T.ivector('token')
		l_in = lasagne.layers.InputLayer(
			shape=(None,), input_var=input_tokens
		)
		l_emb = lasagne.layers.EmbeddingLayer(
			incoming=l_in,
			input_size=self.query_vocabulary_size,
			output_size=self.num_embedding_dimensions,
			W=self.l_embed_query.W
		)
		embedding = get_output(l_emb)
		self._embed = function([input_tokens], embedding)


	def get_param_values(self):
		'''
		returns a list of the trainable parameters *values*, as
		np.ndarray's.
		'''
		return (
			get_all_param_values(self.l_embed_query, trainable=True) +
			get_all_param_values(self.l_embed_context, trainable=True)
		)


	def get_params(self):
		'''returns a list of the trainable parameters, that is, the query
		and context embeddings.  (similar to layer.get_all_params.)'''
		return (
			self.l_embed_query.get_params(trainable=True) +
			self.l_embed_context.get_params(trainable=True)
		)


	def get_output(self):
		'''returns the symbolic output.  (similar to layer.get_output.)'''
		return self.output


	def save(self, directory):
		'''
		Saves the model parameters (embeddings) to disk, in a file called
		"embeddings.npz" under the directory given.
		'''

		# We are willing to create the directory given if it doesn't exist
		if not os.path.exists(directory):
			os.mkdir(directory)

		# Save under the directory given in a file called "embeddings.npz'
		save_path = os.path.join(directory, "embeddings.npz")

		# Get the parameters and save them to disk
		W, C = self.get_param_values()
		np.savez(save_path, W=W, C=C)


	def load(self, directory):
		'''
		Loads the model parameter values (embeddings) stored in the
		directory given.  Expects to find the parameters in a file called
		"embeddings.npz" within the directory given.
		'''

		# By default, we save to a file called "embeddings.npz" within the
		# directory given to the save function.
		save_path = os.path.join(directory, "embeddings.npz")

		# Load the parameters
		npfile = np.load(save_path)
		W_loaded = npfile['W'].astype('float32')
		C_loaded = npfile['C'].astype('float32')

		# Get the theano shared variables that are used to hold the
		# parameters, and fill them with the values just loaded
		W_shared, C_shared = self.get_params()
		W_shared.set_value(W_loaded)
		C_shared.set_value(C_loaded)
