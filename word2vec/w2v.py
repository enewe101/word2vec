from noise_contrast import get_noise_contrastive_loss
import os
import numpy as np
from theano import tensor as T, function, shared
import lasagne
from lasagne.layers import (
	get_output, InputLayer, EmbeddingLayer, get_all_params,
	get_all_param_values
)
from lasagne.init import Normal
from lasagne.updates import nesterov_momentum
from new_minibatcher import DatasetReader, TheanoMinibatcher


def row_dot(matrixA, matrixB):
	C = (matrixA * matrixB).sum(axis=1)
	return C


def sigmoid(tensor_var):
	return 1/(1+T.exp(-tensor_var))


def word2vec(
		files=[],
		directories=[],
		skip=[],
		savedir=None,
		num_epochs=5,
		unigram_dictionary=None,
		noise_ratio=15,
		kernel=[1,2,3,4,5,5,4,3,2,1],
		t = 1.0e-5,
		batch_size = 1000,  # Number of *signal* examples per batch
		num_embedding_dimensions=500,
		word_embedding_init=Normal(),
		context_embedding_init=Normal(),
		learning_rate=0.1,
		momentum=0.9,
		verbose=True,
		num_processes=3
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
	dataset_reader = DatasetReader(
		files=files,
		directories=directories,
		skip=skip,
		batch_size=batch_size, # number of *signal_examples* per batch
		noise_ratio=noise_ratio,
		t=t,
		num_processes=num_processes,
		unigram_dictionary=unigram_dictionary,
		kernel=kernel,
		verbose=verbose
	)

	# Prpare the minibatch generator
	# (this produces the counter_sampler stats)
	print 'Generating dictionary'
	dataset_reader.prepare(savedir=savedir)

	# Make a symbolic minibatcher
	# Note that the full batch includes noise_ratio number of noise examples#
	# for every signal example, and the parameter "batch_size" here is interpreted
	# as just the number of signal examples per batch; the full batch size is:
	full_batch_size = batch_size * (1 + noise_ratio)
	minibatcher = TheanoMinibatcher(
		batch_size = full_batch_size,
		dtype="int32",
		num_dims=2
	)

	# Make a Word2VecEmbedder object, feed it the combined input
	embedder = Word2VecEmbedder(
		input_var=minibatcher.get_batch(),
		batch_size=full_batch_size,
		vocabulary_size=dataset_reader.get_vocab_size(),
		num_embedding_dimensions=num_embedding_dimensions,
		word_embedding_init=word_embedding_init,
		context_embedding_init=context_embedding_init
	)

	# Architectue is ready.  Make the loss function, and use it to create the
	# parameter updates responsible for learning
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

	print 'Generating dataset'
	dataset = dataset_reader.generate_dataset()
	print dataset
	print 'Loading dataset'
	# TODO: freeze the dataset (generate should take a freeze_dir parameter)
	minibatcher.load_dataset(dataset)

	#f = function([], embedder.get_output(), updates=minibatcher.get_updates())
	#for batch_num in range(minibatcher.get_num_batches()):
	#	print f()

	# Iterate over the corpus, training the embeddings
	for epoch in range(num_epochs):
		if verbose:
			print 'starting epoch %d' % epoch
		losses = []
		minibatcher.reset()
		for batch_num in range(minibatcher.get_num_batches()):
			losses.append(train())
		if verbose:
			print '\tAverage loss: %f' % np.mean(losses)

	# Save the model (the embeddings) if savedir was provided
	if savedir is not None:
		embedings_filename = os.path.join(savedir, 'embeddings.npz')
		embedder.save(embeddings_filename)

	# Return the trained embedder and the dictionary mapping tokens
	# to ids
	return embedder, dataset_reader.unigram_dictionary



class Word2VecEmbedder(object):

	def __init__(
		self,
		input_var,
		batch_size,
		vocabulary_size=100000,
		num_embedding_dimensions=500,
		word_embedding_init=Normal(),
		context_embedding_init=Normal(),
	):

		self.input_var = input_var
		self.batch_size = batch_size
		self.vocabulary_size = vocabulary_size
		self.num_embedding_dimensions = num_embedding_dimensions
		self._embed = None

		# Every row (example) in a batch consists of a query word and a
		# context word.  We need to learn separate embeddings for each,
		# and so, the input will be separated into all queries and all
		# contexts, and handled separately.
		self.query_input = input_var[:,0]
		self.context_input = input_var[:,1]

		# Make separate input layers for query and context words
		#print 'using batch size:', batch_size
		self.l_in_query = lasagne.layers.InputLayer(
			shape=(batch_size,), input_var=self.query_input
		)
		self.l_in_context = lasagne.layers.InputLayer(
			shape=(batch_size,), input_var=self.context_input
		)

		# Make separate embedding layers for query and context words
		self.l_embed_query = lasagne.layers.EmbeddingLayer(
			incoming=self.l_in_query,
			input_size=vocabulary_size,
			output_size=num_embedding_dimensions,
			W=word_embedding_init
		)
		self.l_embed_context = lasagne.layers.EmbeddingLayer(
			incoming=self.l_in_context,
			input_size=vocabulary_size,
			output_size=num_embedding_dimensions,
			W=context_embedding_init
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
			input_size=self.vocabulary_size,
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
			get_all_params(self.l_embed_query, trainable=True) +
			get_all_params(self.l_embed_context, trainable=True)
		)


	def get_output(self):
		'''returns the symbolic output.  (similar to layer.get_output.)'''
		return self.output


	def save(self, filename):
		W, C = self.get_param_values()
		np.savez(filename, W=W, C=C)


	def load(self, filename):
		npfile = np.load(filename)
		W_loaded, C_loaded = npfile['W'].astype('float32'), npfile['C'].astype('float32')

		# Get the theano shared variables that are used to hold the
		# parameters, and fill them with the values just loaded
		W_shared, C_shared = self.get_params()
		W_shared.set_value(W_loaded)
		C_shared.set_value(C_loaded)
