from unittest import main, TestCase
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
from lasagne.layers import get_output, InputLayer, EmbeddingLayer
from lasagne.init import Normal
from word2vec import WordEmbedder, NoiseContrastiveEmbedder

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

def sigma(a):
	return 1/(1+np.exp(-a))
usigma = np.vectorize(sigma)



class TestEmbedder(TestCase):


	def setUp(self):

		# Make a query input vector.  This holds indices that represent
		# words in the vocabulary.  For the test we have just three words
		# in the vocabulary
		self.TEST_QUERY_INPUT = np.array([0,2]).astype('int32')

		# Make a test context input.  This also holds indices which identify
		# words in the vocabluary, indices representing 
		self.TEST_CONTEXT_INPUT = np.array(
			[ [(i+j)%V for j in range(6)] for i in range(2) ]
		).astype('int32')

		# Artificially adopt this word embedding matrix for query words
		self.WORD_EMBEDDING_MATRIX = np.array([
			[-0.04576914, -0.26519672, -0.06857708, -0.23748968,  
				0.37556829],
			[ 0.08540803,  0.32099229, -0.19136694, -0.48263541,  
				0.03355749],
			[-0.33319689,  0.26062664,  0.06826347, -0.39083191, 
				-0.31544908]
		])

		# Artificially adopt this word embedding matrix for context words
		self.CONTEXT_EMBEDDING_MATRIX = np.array([
			[-0.29474795, -0.2559814 , -0.04503929,  0.35159791,  
				0.46856225],
			[ 0.00963128,  0.22368461,  0.44933862,  0.48584304, 
				-0.03599219],
			[ 0.05338832, -0.22895403, -0.08288041, -0.47226618, 
				-0.35951928]
		])

		# We'll try embeddings with different dimensions and vocab size too
		self.REDUCED_WORD_EMBEDDING_MATRIX = (
			self.WORD_EMBEDDING_MATRIX.reshape(5,3)
		)
		self.REDUCED_CONTEXT_EMBEDDING_MATRIX = (
			self.CONTEXT_EMBEDDING_MATRIX.reshape(5,3)
		)

		# This is the expected result if we use the TEST_QUERY_INPUT,
		# and TEST_CONTEXT_INPUT as inputs and if we use, as 
		# initializations for the embedding weights, 
		#the WORD_EMBEDDING_MATRIX and CONTEXT_EMBEDDING_MATRIX
		self.EXPECTED_DOTS = np.array([
			[ 0.1769407 , -0.2194758 ,  0.04109232,  0.1769407 , 
				-0.2194758 , 0.04109232],
			[-0.09276679,  0.21486867, -0.25680422, -0.09276679,  
				0.21486867, -0.25680422]
		])
 
		# This is the expected result if we use the TEST_QUERY_INPUT
		# and TEST_CONTEXT_INPUT after modifying word_embedding matrix
		# by adding 10 elementwise
		self.ALT_EXPECTED_DOTS = np.array([
			[2.4208559 ,  11.1055778 , -10.86122348,   2.4208559 ,
				11.1055778 , -10.86122348],
			[11.23228681, -10.68744713,   1.98711098,  11.23228681,
				-10.68744713,   1.98711098]
		])


	def test_learning(self):
		# Declare variable for query input 
		# (the focal word for the skip gram)
		query_input = T.ivector('query_input')

		# Declare vars for context words, one for signal, one for noise
		context_input = T.imatrix('context_input')
		noise_input = T.imatrix('noise_input')

		# Make a noise contrastive embedder object, don't specifically
		# initialize embedding parameters so they are random
		# print 'making embedder'
		vocab_size = 10
		signal_contexts_per_word = 6
		noise_contexts_per_word = 60
		num_embedding_dimensions = 5
		noise_contrastive_embedder = NoiseContrastiveEmbedder(
			query_input,
			context_input,
			noise_input,
			context_shape=(None,signal_contexts_per_word),
			noise_shape=(None,noise_contexts_per_word),
			vocabulary_size=vocab_size,
			num_embedding_dimensions=num_embedding_dimensions
		)

		# print 'making loss expression and sgd updates'
		loss = noise_contrastive_embedder.get_loss()
		params = noise_contrastive_embedder.get_all_params()
		learning_rate = 0.1
		updates = lasagne.updates.sgd(loss, params, learning_rate)
		updates = lasagne.updates.nesterov_momentum(
			loss, params, learning_rate, 0.9)

		batch_words = np.array([0,1,2,3,4,5,6,7,8,9]).astype('int32')
		batch_contexts = np.array([
			[2,2,2,2,2,2], # context for 0
			[3,3,3,3,3,3], # context for 1
			[0,0,0,0,0,0], # context for 2
			[1,1,1,1,1,1], # context for 3

			[6,6,6,6,6,6], # context for 4
			[7,7,7,7,7,7], # context for 5
			[4,4,4,4,4,4], # context for 6
			[5,5,5,5,5,5], # context for 7

			[9,9,9,9,9,9], # context for 8
			[8,8,8,8,8,8]  # context for 9
		]).astype('int32')

		batch_noise = np.random.randint(
			0, vocab_size,(len(batch_contexts),noise_contexts_per_word)
		).astype('int32')

		# print 'compiling the training function'

		training_fn = function(
			[query_input, context_input, noise_input],
			loss, updates=updates
		)

		num_epochs = 1500
		num_replicates = 10
		W, C = noise_contrastive_embedder.get_all_params()
		embedding_products = []
		for replicate in range(num_replicates):

			embedding_shape = (vocab_size, num_embedding_dimensions)
			W.set_value(np.random.normal(0, 0.01, embedding_shape))
			C.set_value(np.random.normal(0, 0.01, embedding_shape))

			for epoch in range(num_epochs):

				batch_noise = np.random.randint(
					0, vocab_size,
					(len(batch_contexts),noise_contexts_per_word)
				).astype('int32')

				training_fn(
					batch_words, batch_contexts, batch_noise
				)

			embedding_product = np.dot(W.get_value(), C.get_value().T)
			embedding_products.append(usigma(embedding_product))
		
		mean_embedding_products = np.mean(embedding_products, axis=0)

		# We expect that the embeddings will allocate the most probability
		# to the contexts that were provided for words in the toy data.
		# We always provided a single context via batch_contexts 
		# (e.g. context 2 provided for word 0), so we expect these contexts
		# to be the maximum.
		expected_max_prob_contexts = batch_contexts[:,0]
		self.assertTrue(np.array_equal(
			np.argmax(mean_embedding_products, axis=1),
			expected_max_prob_contexts
		))
			
		# The dot product of a given word embedding and context embedding
		# have an interpretation as the probability that that word and
		# context derived from the toy data instead of the noise.
		# See equation 3 in Noise-Contrastive Estimation of Unnormalized 
		# Statistical Models, with Applications to Natural Image 
		# StatisticsJournal of Machine Learning Research 13 (2012), 
		# pp.307-361.
		# That shows the probability should be around 0.5
		# Since the actual values are stocastic, we check that the 
		# average of repeated trials is within 0.25 - 0.75.
		embedding_maxima = np.max(mean_embedding_products, axis=1)
		self.assertTrue(all(
			[x > 0.25 for x in embedding_maxima]
		))
		self.assertTrue(all(
			[x < 0.75 for x in embedding_maxima]
		))


	# TODO: add assertions
	def test_get_signal_score(self):
		'''
		Tests get_signal_score.  It should return elementwise sigmoid
		applied to each element in the signal_output, followed by summation
		along the first axis (that is, summing the result obtained for
		all contexts applied to a given word), and then with the mean
		taken for all word examples.
		'''

		# Declare variable for query input 
		# (the focal word for the skip gram)
		query_input = T.ivector('query_input')

		# Declare vars for context words, one for signal, one for noise
		context_input = T.imatrix('context_input')
		noise_input = T.imatrix('noise_input')


		# Make a noise contrastive embedder object
		noise_contrastive_embedder = NoiseContrastiveEmbedder(
			query_input,
			context_input,
			noise_input,
			context_shape=(None,K+1),
			noise_shape=(None,K+1),
			vocabulary_size=V,
			num_embedding_dimensions=D,
			word_embedding_init=self.WORD_EMBEDDING_MATRIX,
			context_embedding_init=self.CONTEXT_EMBEDDING_MATRIX
		)

		signal_score = noise_contrastive_embedder.get_signal_score()
		noise_score = noise_contrastive_embedder.get_noise_score()

		f = function([query_input, context_input], signal_score)
		g = function([query_input, noise_input], noise_score)

		signal_score_result = f(
			self.TEST_QUERY_INPUT, self.TEST_CONTEXT_INPUT
		)
		#print signal_score_result

		# We're using the same vector to test the noise function, this
		# is just convenient.
		test_noise_input = self.TEST_CONTEXT_INPUT
		noise_score_result = g(self.TEST_QUERY_INPUT, test_noise_input)
		#print noise_score_result


	def test_NoiseContrastiveEmbedder(self):

		'''
		Tests that the two channels of the noise contrastive embedder ---
		the signal channel and the noise channel --- have the same 
		architecture, with shared parameters.  This is done by passing the 
		same values for the signal context and the noise context words and 
		checking that the results agree.  Then the shared parameters are 
		modified, and we check that the output from the signal and noise 
		channels have changed to new values, but still agree with one 
		another.
		'''

		# Declare variable for query input 
		# (the focal word for the skip gram)
		query_input = T.ivector('query_input')

		# Declare vars for context words, one for signal, one for noise
		context_input = T.imatrix('context_input')
		noise_input = T.imatrix('noise_input')

		# Make a noise contrastive embedder object
		noise_contrastive_embedder = NoiseContrastiveEmbedder(
			query_input,
			context_input,
			noise_input,
			context_shape=(None,K+1),
			noise_shape=(None,K+1),
			vocabulary_size=V,
			num_embedding_dimensions=D,
			word_embedding_init=self.WORD_EMBEDDING_MATRIX,
			context_embedding_init=self.CONTEXT_EMBEDDING_MATRIX
		)

		# Get the output from the signal and noise channels
		signal_match = noise_contrastive_embedder.get_signal_output()
		noise_match = noise_contrastive_embedder.get_noise_output()

		# Compile functions representing forward pass through the channels
		f = function([query_input, context_input], signal_match)
		g = function([query_input, noise_input], noise_match)

		# Compute the outputs through both channels.  
		# They should be the same.  Print them
		test_signal_match = f(
			self.TEST_QUERY_INPUT, self.TEST_CONTEXT_INPUT)
		test_noise_match = g(
			self.TEST_QUERY_INPUT, self.TEST_CONTEXT_INPUT)
		#print 'test signal match:'
		#print test_signal_match
		#print 'test noise match:'
		#print test_noise_match

		self.assertTrue(np.allclose(test_signal_match, self.EXPECTED_DOTS))
		self.assertTrue(np.allclose(test_signal_match, test_noise_match))

		# Now change the word embedding
		word_embedding_matrix = self.WORD_EMBEDDING_MATRIX + 10
		w_emb = (
			noise_contrastive_embedder.signal_architecture.l_embed_query.W
		)
		w_emb.set_value(word_embedding_matrix)

		# Re-compute the outputs through both channels.
		new_signal_match = f(self.TEST_QUERY_INPUT, self.TEST_CONTEXT_INPUT)
		new_noise_match = g(self.TEST_QUERY_INPUT, self.TEST_CONTEXT_INPUT)
		#print 'new signal match'
		#print repr(new_signal_match)
		#print 'new noise match'
		#print repr(new_noise_match)

		# Check that they are what is expected, and that they are still
		# identical to one another
		self.assertTrue(np.allclose(
			new_signal_match, self.ALT_EXPECTED_DOTS))
		self.assertTrue(np.allclose(new_signal_match, new_noise_match))


	def test_WordEmbedder(self):

		query_input = T.ivector('query_input')
		context_input = T.imatrix('context_input')

		embedder = WordEmbedder(
			query_input,
			context_input,
			context_shape=(None, K+1),	# (num_examples, num_contexts_per)
			vocabulary_size=V,
			num_embedding_dimensions=D,
			word_embedding_init=self.WORD_EMBEDDING_MATRIX,
			context_embedding_init=self.CONTEXT_EMBEDDING_MATRIX
		)

		query_embedding = embedder.get_word_embedding()
		context_embedding = embedder.get_context_embedding()
		dots = embedder.get_output()

		f = function([query_input], query_embedding)
		g = function([context_input], context_embedding)
		h = function([query_input, context_input], dots)
		
		#print 'query embeddings:'
		#print f(self.TEST_QUERY_INPUT)
		#print
		#print 'context embeddings:'
		#print g(self.TEST_CONTEXT_INPUT)
		#print 
		test_output = h(self.TEST_QUERY_INPUT, self.TEST_CONTEXT_INPUT)
		#print 'output:'
		#print repr(test_output)
		expected = np.array([
			[0., 10., 20., 0., 10., 20.],
			[60., 120., 0., 60., 120., 0.]
		])
		self.assertTrue(np.allclose(test_output, self.EXPECTED_DOTS))
		#print
		#print 'passed!'


if __name__=='__main__': 
	main()
	
