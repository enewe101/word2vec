import os
import sys
import numpy as np
import theano
import theano.tensor as T
from theano import function, scan
import lasagne
from lasagne.layers import (
	get_output, InputLayer, EmbeddingLayer, get_all_params,
	get_all_param_values
)
from lasagne.init import Normal
from dictionary import Dictionary
from unigram import Unigram, MultinomialSampler


def word2vec(
		files=[],
		directories=[],

		# ^^^ At a minimum, either files or directories should be specified
		# vvv All other arguments are strictly optional

		savedir=None,
		num_epochs=5,
		skip=[],
		dictionary=None,
		unigram=None,
		noise_ratio=15,
		kernel=[1,2,3,4,5,5,4,3,2,1],
		t = 1.0e-5,
		batch_size = 1000,
		num_embedding_dimensions=500,
		word_embedding_init=Normal(), 
		context_embedding_init=Normal(),
		learning_rate=0.1,
		momentum=0.9,
		verbose=True,
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

	# Define the input theano variables
	signal_input = T.imatrix('query_input')
	noise_input = T.imatrix('noise_input')

	# Make a NoiseContraster, and get the combined input
	noise_contraster = NoiseContraster(signal_input, noise_input)
	combined_input = noise_contraster.get_combined_input()

	# Make a Word2VecEmbedder object, feed it the combined input
	word2vec_embedder = Word2VecEmbedder(
		combined_input,
		batch_size,
		vocab_size,
		num_embedding_dimensions
	)

	# Get the params and output from the word2vec embedder, feed that
	# back to the noise_contraster to get the training function
	combined_output = word2vec_embedder.get_output()
	params = word2vec_embedder.get_params()
	train = noise_contraster.get_train_func(combined_output, params)

	# Make a MinibatchGenerator
	minibatch_generator = MinibatchGenerator(
		files=files,
		directories=directories,
		skip=skip,
		dictionary=dictionary,
		unigram=unigram,
		noise_ratio=noise_ratio,
		kernel=kernel,
		t=t,
		batch_size=batch_size,
		parse=parse
	)

	# Prpare the minibatch generator (this produces the unigram stats)
	minibatch_generator.prepare(savedir=savedir)

	# Iterate over the corpus, training the embeddings
	for epoch in range(num_epochs):
		print 'starting epoch %d' % epoch
		for signal_batch, noise_batch in minibatch_generator.generate():
			loss = train(signal_batch, noise_batch)

	# Save the model (the embeddings) if savedir was provided
	if savedir is not None:
		embedings_filename = os.path.join(savedir, 'embeddings.npz')
		word2vec_embedder.save(embeddings_filename)

	# Return the trained word2vec_embedder
	return word2vec_embedder









def noise_contrast(signal, noise, scale=True):
	'''
	Takes the theano symbolic variables `signal` and `noise`, whose
	elements are interpreted as probabilities, and creates a loss
	function which rewards large probabilities in signal and penalizes 
	large probabilities in noise.
	
	`signal` and `noise` can have any shape.  Their contributions will be 
	summed over all dimensions.

	If `scale` is true, then scale the loss function by the size of the
	signal tensor --- i.e. divide by the signal batch size.  This makes
	the scale of the loss function invariant to changes in batch size
	'''

	signal_score = T.log(signal).sum()
	noise_score = T.log(1-noise).sum()
	objective = signal_score + noise_score
	loss = -objective

	loss = loss / signal.shape[0]
	
	return loss



class NoiseContraster(object):

	def __init__(
		self,
		signal_input,
		noise_input,
		learning_rate=0.1,
		momentum=0.9
	):
		self.signal_input = signal_input
		self.noise_input = noise_input
		self.learning_rate = learning_rate
		self.momentum=momentum

		self.combined = T.concatenate([
			self.signal_input, self.noise_input
		])

	def get_combined_input(self):
		return self.combined

	def get_train_func(self, output, params):

		# Split the output based on the size of the individual input streams
		self.signal_output = output[:self.signal_input.shape[0]]
		self.noise_output = output[self.signal_input.shape[0]:]

		# Construct the loss based on Noise Contrastive Estimation
		self.loss = noise_contrast(self.signal_output, self.noise_output)

		# Get the parameter updates using stochastic gradient descent with
		# nesterov momentum.  TODO: make the updates configurable
		self.updates = lasagne.updates.nesterov_momentum(
			self.loss,
			params,
			self.learning_rate,
			self.momentum
		)

		self.train = function(
			inputs=[self.signal_input, self.noise_input], 
			outputs=self.loss,
			updates=self.updates
		)

		return self.train



class Word2Vec(object):

	def __init__(
		self,
		batch_size,
		vocabulary_size=100000,
		num_embedding_dimensions=500,
		word_embedding_init=Normal(), 
		context_embedding_init=Normal(),
		learning_rate=0.1,
		momentum=0.9,
		verbose=True
	):

		# Register all the arguments for easy inspection and access
		self.batch_size = batch_size
		self.vocabulary_size = vocabulary_size
		self.num_embedding_dimensions = num_embedding_dimensions
		self.word_embedding_init = word_embedding_init
		self.context_embedding_init = context_embedding_init
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.verbose = verbose

		# The input that we pass through the Word2VecEmbedder will be
		# the concatenation of the signal and noise examples
		self.positive_input = T.imatrix('positive_input')
		self.negative_input = T.imatrix('negative_input')
		embedder_input = T.concatenate([
			self.positive_input, self.negative_input
		])

		# Make a Word2VecEmbedder
		self.embedder = Word2VecEmbedder(
			embedder_input,
			batch_size,
			vocabulary_size=vocabulary_size,
			num_embedding_dimensions=num_embedding_dimensions,
			word_embedding_init=word_embedding_init,
			context_embedding_init=context_embedding_init
		)

		# Split the output back up into its positive and negative parts
		embedder_output = self.embedder.get_output()

		self.positive_output = embedder_output[
			:self.positive_input.shape[0]
		]
		self.negative_output = embedder_output[
			self.positive_input.shape[0]:
		]

		# Construct the loss based on Noise Contrastive Estimation
		self.loss = noise_contrast(
			self.positive_output, self.negative_output
		)

		# Get the parameter updates using stochastic gradient descent with
		# nesterov momentum.  TODO: make the updates configurable
		self.updates = lasagne.updates.nesterov_momentum(
			self.loss,
			self.embedder.get_params(), 
			learning_rate,
			momentum
		)

		# That's everything we need to compile a training function.
		# The training fuction will be defined separately, and will 
		# wrap the compiled theano function, which will be compiled when
		# it is first called.


	def get_param_values(self):
		'''
		Returns the values of the embedding parameters, from the 
		underlying embedder.
		'''
		return self.embedder.get_param_values()

	def get_params(self):
		'''
		Returns the embedding parameters, of the underlying embedders.
		'''
		return self.embedder.get_params()


	def train_from_corpus(
		self, files=[], directories=[], skip=[],
		loaddir=None, savedir=None
	):
		'''
		Takes an iterable of paths that point to corpus files / folders.
		Reads those files, and:
		0) Reads sentence-split, tokenized documents in a way that 
			manages memory and avoid's blocking downstream steps during
			I/O requests.
			- a reading process stays ahead of the downstream process in
				loading files into memory
			- a customizable parser is responsible for understanding the
				file format
			- `files` and `directories` can be strings, or iterables of
				strings, including custom generators that walk the 
				filesystem
		1) Makes a dictionary which encodes tokens as integers
			- the dictionary class stores a two-way mapping between 
				token types (i.e. unique words) and integers
			- forward mapping from tokens to integers uses a dictionary
				with keys as tokens and ints as values
			- revrese mapping from integers to tokens uses a list of tokens
		2) Makes a unigram model which enables sampling from the unigram
			model
			- keeps count of how many times a given tokens were seen
				(using a list whose values are counts and whose 
					components are the token ids)
			- enables rapid sampling from the unigram distribution. Using
				a binary tree structure.  The first time sampling is 
				attempted after an update has been made to the counts,
				the tree is updated / generated.
			- a full pass through the corpus is needed to create the unigram
				model.
		3) Generates batches of signal and noise examples to train the
			embeddings


		If `savedir` is specified, then three files will be saved into
		the directory specified by savedir (savedir will be created if 
		it doesn't exist, as long as other dirs in its path already exist)
		1) savedir/dictionary.gz -- stores the dictionary mapping
		2) savedir/unigram.gz -- stores the unigram frequencies.  This 
			means that future training using different sampling will
			not need to re-count the unigram frequencies!
		3) savedir/embeddings.gz -- stores the embedding parameters for
			both query-embeddings (which is the main word embedding of use
			in other applications) as well as the context-embedding (not
			usually needed for other applications, but kept for 
			completeness)

		if `loaddir` is specified, the dictionary and unigram saved in
		loaddir/dictionary.gz and loaddir/unigram.gz will be loaded.
		This means that the unigram frequencies (and dictionary) don't 
		need to be made before training.
		'''

		# If a savedir is given, check that it is valid before proceeding
		# that way we fail fast if there's an IO issue
		if savedir is not None:

			# if savedir exists, just make sure it's not a file
			if os.path.exsits(savedir):
				if os.path.isfile(savedir):
					raise IOError(
						'Path supplied as `savedir` is a file: %s'
						% savedir
					)

			# if savedir doesn't exist, try to make it
			else:
				os.mkdir(savedir)

		# Create a dictionary.  We'll pass it into the minibatch generator,
		# but we want a reference to it in the Word2Vec object too, 
		# since it also needs the dictionary
		dictionary = Dictionary()

		# Make a minibatch generator
		minibatch_generator = MinibatchGenerator(
			files, directories, skip, dictionary
		)

		# Run the minibatch generator over the corpus to collect unigram
		# statistics and to fill out the dictionary
		minibatch_generator.prepare_unigram()

		# We now have a full dictionary and the unigram statistics.
		# Save them if savedir was specified
		if savedir is not None:
			dictionary.save(os.path.join(savedir, 'dictionary.gz'))
			minibatch_generator.save_unigram(
				os.path.join(savedir, 'unigram.gz')
			)

		# Here is where the training actually happens
		for positive_input, negative_input in minibatch_generator:
			self.train(positive_input, negative_input)

		# Save the learned embedding (if savedir was specified)
		if savedir is not None:
			self.embedder.save(os.path.join(savedir, 'embedding.npz'))


	def save(self, savedir):
		self.dictionary.save(os.path.join(savedir, 'dictionary.gz'))
		self.embedder.save(os.path.join(savedir, 'embedding.npz'))


	def train(self, positive_input, negative_input):
		'''
		Runs the compiled theano training function, using the positive
		query-context pairs contained in positive_input and the negative
		query_context pairs contained in negative_input.  Both are expected
		to be numpy int32-type arrays, with shape (batch_size, 2), each
		example is a row with two elements: the query-word and the
		context-word, coded as integers.
  
		If the theano training function hasn't been compiled yet (which
		is true the first time this is called, it compiles it.
		'''
		if not hasattr(self, 'train_'):
			if self.verbose:
				print 'Compiling training function on first call.'
			self.train_ = function(
				inputs=[self.positive_input, self.negative_input], 
				outputs=self.loss,
				updates=self.updates
			)

		return self.train_(positive_input, negative_input)






def row_dot(matrixA, matrixB):
	C = (matrixA * matrixB).sum(axis=1)
	return C


#def row_dot(matrixA, matrixB):
#	C, updates = theano.scan(
#		fn=lambda x,y: T.dot(x,y),
#		outputs_info=None,
#		sequences=[matrixA, matrixB]
#	)
#	return C


def sigmoid(tensor_var):
	return 1/(1+T.exp(-tensor_var))


class Word2VecEmbedder(object):

	def __init__(
		self,
		input_var,
		batch_size,
		vocabulary_size=100000,
		num_embedding_dimensions=500,
		word_embedding_init=Normal(),
		context_embedding_init=Normal(),
		dropout=0.
	):

		self.input_var = input_var

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
		# We can express this as matrix multiplication: if we multiply
		# the query embeddings (a matrix) with the transposed context
		# embeddings (also a matrix), the elements of the result give
		# the dot prodcuts of all context embeddings with all query
		# embeddings.  We only want the dot products for the queries
		# and contexts that came from the same example (not the dot products
		# formed by all pairs), which we can obtain by selecting the 
		# diagonal from the result.  Hopefully theano's optimizations are
		# smart enough to only calculate the needed dot products.
		self.match_scores = row_dot(
			self.query_embedding,
			self.context_embedding
		)
		#self.match_scores = T.dot(
		#	self.query_embedding, self.context_embedding.T
		#).diagonal()

		# Finally apply the sigmoid activation function
		self.output = sigmoid(self.match_scores)


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
		W_loaded, C_loaded = npfile['W'], npfile['C']

		# Get the theano shared variables that are used to hold the 
		# parameters, and fill them with the values just loaded
		W_shared, C_shared = self.get_params()
		W_shared.set_value(W_loaded)
		C_shared.set_value(C_loaded)


