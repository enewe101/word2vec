import os
import sys
import theano
import time
import theano.tensor as T
from theano import function, scan
import lasagne
import numpy as np
from lasagne_mnist import build_mlp
from mnist_iterator import MnistIterator
from lasagne.layers import (
	get_output, InputLayer, EmbeddingLayer, get_all_params
)
from lasagne.init import Normal

# Number of noise context words to true context words
K = 15
# Size of vocabulary for embedding
V = 100000
# Embedding dimensionality
D = 500

# Debug
D = 5
V = 3
K = 5


class NoiseContrastiveEmbedder(object):

	def __init__(
		self,
		query_input,
		context_input,
		noise_input,
		context_shape=(None,None),
		noise_shape=(None,None),
		vocabulary_size=100000,
		num_embedding_dimensions=500,
		word_embedding_init=Normal(), 
		context_embedding_init=Normal()
	):

		# Keep a reference to the symbolic inputs
		self.query_input = query_input
		self.context_input = context_input
		self.noise_input = noise_input

		# Make a word embedder to process signal context (we'll make one to
		# process noise context below)
		self.signal_architecture = WordEmbedder(
			query_input,
			context_input,
			context_shape=context_shape,
			vocabulary_size=vocabulary_size,
			num_embedding_dimensions=num_embedding_dimensions,
			word_embedding_init=word_embedding_init,
			context_embedding_init=context_embedding_init
		)

		# Get shorter names for the embedding parameters.  We'll share
		# the parameters with the word embedder for noise context.
		query_embed_params, context_embed_params = (
			self.signal_architecture.get_all_params()
		)

		# Make a workd embedder to process noise context (pass in embedding
		# parameters from the signal context embedder.
		self.noise_architecture = WordEmbedder(
			query_input, 
			noise_input,
			context_shape=noise_shape,
			vocabulary_size=vocabulary_size,
			num_embedding_dimensions=num_embedding_dimensions,
			word_embedding_init=query_embed_params,
			context_embedding_init=context_embed_params
		)


	def get_signal_output(self):
		return self.signal_architecture.get_output()


	def get_noise_output(self):
		return self.noise_architecture.get_output()


	def get_signal_activation(self):
		activation = 1 / (1 + T.exp(-self.get_signal_output()))
		return activation


	def get_noise_activation(self):
		activation = 1 / (1 + T.exp(self.get_noise_output()))
		return activation


	def get_signal_score(self):
		# Compute sigmoid activiation for each word-context dot product
		# then sum for different contexts and average for each word
		activation = self.get_signal_activation()
		log_prob = T.log2(activation)
		return log_prob.sum(axis=1).mean()


	def get_noise_score(self):
		# Similarly for noise, but take signmoid of negative dot product
		activation = self.get_noise_activation()
		log_prob = T.log2(activation)
		return log_prob.sum(axis=1).mean()


	def get_loss(self):
		# Total up the score.  Seeking max, so take negative as loss.
		total_score = self.get_signal_score() + self.get_noise_score()

		loss = -total_score

		return loss


	def get_all_params(self):
		return self.signal_architecture.get_all_params()


	def embed_context(self, words_to_embed):
		# Expose the context embedding function of the signal architecture
		return self.signal_architecture.embed_context(words_to_embed)


	def embed_words(self, words_to_embed):
		# Expose the word embedding function of the signal architecture
		return self.signal_architecture.embed_words(words_to_embed)


class WordEmbedder(object):

	def __init__(
		self,
		query_input,
		context_input,
		context_shape=(None, None),	# (num_examples, num_contexts_per_word)
		vocabulary_size=100000,
		num_embedding_dimensions=500,
		word_embedding_init=Normal(),
		context_embedding_init=Normal(),
		dropout=0.
	):

		try:
			num_examples, num_contexts_per_word = context_shape
		except ValueError:
			print context_shape

		self.query_input = query_input

		# TODO: this should be able to be None
		self.context_input = context_input

		# Make an input layer for query words
		self.l_in_query = lasagne.layers.InputLayer(
			shape=(num_examples,), input_var=query_input
		)

		# Make an embedding layer for context words
		self.l_embed_query = lasagne.layers.EmbeddingLayer(
			incoming=self.l_in_query,
			input_size=vocabulary_size, 
			output_size=num_embedding_dimensions, 
			W=word_embedding_init
		)
		if dropout > 0:
			print 'using dropout'
			self.l_drop_query = lasagne.layers.DropoutLayer(
				self.l_embed_query, p=dropout
			)
			self.query_embedding = get_output(self.l_drop_query)
		else:
			self.query_embedding = get_output(self.l_embed_query)

		# Make an input layer for context words.  It's a matrix because
		# we expect multiple context words per query word.
		self.l_in_context = lasagne.layers.InputLayer(
			shape=(num_examples,num_contexts_per_word), 
			input_var=context_input
		)
		self.l_embed_context = lasagne.layers.EmbeddingLayer(
			incoming=self.l_in_context,
			input_size=vocabulary_size, 
			output_size=num_embedding_dimensions, 
			W=context_embedding_init
		)

		if dropout > 0:
			print 'using dropout'
			self.l_drop_context = lasagne.layers.DropoutLayer(
				self.l_embed_context, p=dropout
			)
			self.context_embedding = get_output(self.l_drop_context)
		else:
			self.context_embedding = lasagne.layers.get_output(
				self.l_embed_context)

		# Take the dot product of each query embedding with its 
		# corresponding Noise and context embeddings
		self.dots, scan_updates = scan(
			fn=lambda P, Q: T.dot(P, Q.dimshuffle(1, 0)),
			outputs_info=None,
			sequences=[self.query_embedding, self.context_embedding]
		)

	def get_all_params(self):
		return (
			get_all_params(self.l_embed_query, trainable=True) +
			get_all_params(self.l_embed_context, trainable=True)
		)

	def get_output(self):
		return self.dots

	def get_word_embedding(self):
		return self.query_embedding

	def get_context_embedding(self):
		return self.context_embedding

	def get_embedding_layer(self):
		return self.l_embed_query

	#TODO
	def embed_context(self, words_to_embed):
		# This function expects words_to_embed to have the shape
		# equal to context_shape passed to the WordEmbedder constructor
		if not hasattr(self, 'do_embed_context'):
			self.do_embed_context = function(
				[self.context_input], self.context_embedding
			)

		return self.do_embed_context(words_to_embed)

	def embed_words(self, words_to_embed):
		# This function expects words_to_embed to be a vector of integers,
		# each being the vocabulary index for a word to be embedded
		if not hasattr(self, 'do_embed_word'):
			self.do_embed_word = function(
				[self.query_input], self.query_embedding
			)

		return self.do_embed_word(words_to_embed)



def main():

	query_input = T.ivector('query_input')
	context_input = T.imatrix('context_input')

	word_embedding_init = (
		np.arange(3*5).reshape((3,5)).astype('float32')
	)
	context_embedding_init = (
		np.array([[i, i, i, i, i] for i in range(3)]).astype('float32')
	)


	query_embedding, context_embedding, query_context_dots = build_network(
		query_input,
		context_input,
		context_shape=(None, K+1),	# (num_examples, num_contexts_per_word)
		vocabulary_size=V,
		num_embedding_dimensions=D,
		word_embedding_init=word_embedding_init,
		context_embedding_init=context_embedding_init
	)


	noise_input = T.imatrix('noise_input')
	query_embedding, noise_embedding, query_noise_dots = build_network(
		query_input, context_input
	)

	embed_queries = function([query_input], query_embedding)
	embed_contexts = function([context_input], context_embedding)
	forward_pass = function([query_input, context_input], dots)

	

	#test_query_input = np.array([0,2]).astype('int32')
	#test_context_input = np.array(
	#	[ [(i+j)%V for j in range(6)] for i in range(2) ]
	#).astype('int32')


