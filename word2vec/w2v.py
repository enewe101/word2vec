import os
import sys
import numpy as np
import theano
import theano.tensor as T
from theano import function, scan
import lasagne
from lasagne.layers import (
	get_output, InputLayer, EmbeddingLayer, get_all_params
)
from lasagne.init import Normal


def noise_contrast(signal, noise, scale=True):
	'''
	Takes the theano symbolic variables `signal` and `noise`, whose
	elements are interpreted as storing probabilities, and creates a loss
	function which penalizes large probabilities in noise and small 
	probabilities in signal.  
	
	`signal` and `noise` can have any shape.  If they have more than one
	dimension, their contributions will be summed over all dimensions.

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


class Word2Vec(object):

	def __init__(
		self,
		positive_input,
		negative_input,
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
		self.positive_input = positive_input
		self.negative_input = negative_input
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
		embedder_input = T.concatenate([positive_input, negative_input])

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
		self.positive_output = embedder_output[:positive_input.shape[0]]
		self.negative_output = embedder_output[positive_input.shape[0]:]

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


	def get_params(self):
		'''returns the embedding parameters, of the underlying embedder'''
		return self.embedder.get_params()


	def train(self, positive_input, negative_input):
		'''Runs the compiled theano training function.  If the training
		function hasn't been compiled yet, it compiles it.
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
		self.match_scores = T.dot(
			self.query_embedding, self.context_embedding.T
		).diagonal()

		# Finally apply the sigmoid activation function
		self.output = sigmoid(self.match_scores)


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


