import time
from unittest import main, TestCase
import random
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
from word2vec_simple import Word2VecEmbedder, Word2Vec, noise_contrast


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



class TestNoiseContrast(TestCase):
	def test_noise_contrast(self):
		signal_input = T.dvector()
		noise_input = T.dmatrix()
		loss = noise_contrast(signal_input, noise_input)

		f = function([signal_input, noise_input], loss)

		test_signal = np.array([.6, .7, .8])
		test_noise = np.array([[.2, .3, .1],[.5, .6, .7]])
		test_loss = f(test_signal, test_noise)

		expected_objective = (
			np.log(test_signal).sum() + np.log(1-test_noise).sum()
		)
		expected_loss = -expected_objective / float(len(test_signal))

		#print 'expected_loss:'
		#print expected_loss
		#print
		#print 'test_loss:'
		#print test_loss
		#print

		self.assertAlmostEqual(test_loss, expected_loss)


class TestWord2Vec(TestCase):


	def setUp(self):

		self.VOCAB_SIZE = 3
		self.NUM_EMBEDDING_DIMENSIONS = 4

		# Make a query input vector.  This holds indices that represent
		# words in the vocabulary.  For the test we have just three words
		# in the vocabulary
		self.TEST_INPUT = np.array(
			[[i / self.VOCAB_SIZE, i % self.VOCAB_SIZE] for i in range(9)]
		).astype('int32')

		# Artificially adopt this word embedding matrix for query words
		self.QUERY_EMBEDDING = np.array([
			[-0.04576914, -0.26519672, -0.06857708, -0.23748968],
			[ 0.08540803,  0.32099229, -0.19136694, -0.48263541],  
			[-0.33319689,  0.26062664,  0.06826347, -0.39083191]
		])

		# Artificially adopt this word embedding matrix for context words
		self.CONTEXT_EMBEDDING = np.array([
			[-0.29474795, -0.2559814 , -0.04503929,  0.35159791],
			[ 0.00963128,  0.22368461,  0.44933862,  0.48584304], 
			[ 0.05338832, -0.22895403, -0.08288041, -0.47226618],
		])

		# This is the expected result if we use the TEST_QUERY_INPUT,
		# and TEST_CONTEXT_INPUT as inputs and if we use, as 
		# initializations for the embedding weights, 
		#the QUERY_EMBEDDING and CONTEXT_EMBEDDING
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

		positive_input = T.imatrix('query_input')
		negative_input = T.imatrix('noise_input')

		# Predifine the size of batches and the embedding
		batch_size = 160
		vocab_size = 10
		num_embedding_dimensions = 5

		# Make a Word2Vec object
		word2vec = Word2Vec(
			positive_input,
			negative_input,
			batch_size,
			vocabulary_size=vocab_size,
			num_embedding_dimensions=num_embedding_dimensions,
			learning_rate=0.1,
			verbose=False
		)

		# Make the positive input.  First component of each example is
		# the query input, and second component is the context.  In the 
		# final embeddings that are learned, dotting these rows and columns
		# respectively from the query and context embedding matrices should
		# give higher values than any other row-column dot products.
		test_positive_input = np.array([
			[0,2],
			[1,3],
			[2,0],
			[3,1],
			[4,6],
			[5,7],
			[6,4],
			[7,5],
			[8,9],
			[9,8]
		]).astype('int32')


		
		num_replicates = 5
		num_epochs = 3000
		embedding_products = []
		W, C = word2vec.get_params()
		start = time.time()
		for rep in range(num_replicates):
			W.set_value(np.random.normal(
				0, 0.01, (vocab_size, num_embedding_dimensions)
			))
			C.set_value(np.random.normal(
				0, 0.01, (vocab_size, num_embedding_dimensions)
			))
			#print '\t***'
			for epoch in range(num_epochs):

				# Sample new noise examples every epoch (this is better than
				# fixing the noise once at the start).
				# Provide 15 negative examples for each query word
				test_negative_input = np.array([
					[i / 10, random.randint(0,9)] for i in range(100)
				]).astype('int32')

				loss = word2vec.train(
					test_positive_input, test_negative_input
				)
				#print loss

			embedding_product = np.dot(W.get_value(), C.get_value().T)
			embedding_products.append(usigma(embedding_product))

		mean_embedding_products = np.mean(embedding_products, axis=0)
		#print np.round(mean_embedding_products, 2)

		# We expect that the embeddings will allocate the most probability
		# to the contexts that were provided for words in the toy data.
		# We always provided a single context via batch_contexts 
		# (e.g. context 2 provided for word 0), so we expect these contexts
		# to be the maximum.
		expected_max_prob_contexts = test_positive_input[:,1]
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
		print 'average weight to correct pairs:', np.mean(embedding_maxima)
		print 'elapsed time:', time.time() - start


	def test_positive_negative_separation(self):

		positive_input = T.imatrix('positive_input')
		negative_input = T.imatrix('negative_input')

		input_var = T.imatrix('input_var')

		positive_input,
		negative_input,
		vocabulary_size=100000,
		num_embedding_dimensions=500,
		word_embedding_init=Normal(), 
		context_embedding_init=Normal(),

		embedder = Word2Vec(
			positive_input,
			negative_input,
			batch_size=len(self.TEST_INPUT),
			vocabulary_size=self.VOCAB_SIZE,
			num_embedding_dimensions=self.NUM_EMBEDDING_DIMENSIONS,
			word_embedding_init=self.QUERY_EMBEDDING,
			context_embedding_init=self.CONTEXT_EMBEDDING
		)

		positive_output = embedder.positive_output
		negative_output = embedder.negative_output

		f = function(
			[positive_input, negative_input],
			[positive_output, negative_output]
		)

		# Calculate the positive and negative output
		test_positive_input = self.TEST_INPUT[:6]
		test_negative_input = self.TEST_INPUT[6:]
		test_positive_output, test_negative_output = f(
			test_positive_input, test_negative_input
		)

		# Calculate the expected embeddings and output
		expected_query_embeddings = np.repeat(
			self.QUERY_EMBEDDING, 3, axis=0
		)
		expected_context_embeddings = np.tile(
			self.CONTEXT_EMBEDDING, (3,1)
		)
		expected_output = usigma(np.dot(
			expected_query_embeddings, expected_context_embeddings.T
		)).diagonal()
		expected_positive_output = expected_output[:6]
		expected_negative_output = expected_output[6:]

		#print 'test_positive_output:'
		#print test_positive_output
		#print
		#print 'expected_positive_output:'
		#print expected_positive_output
		#print
		#print 'test_negative_output:'
		#print test_negative_output
		#print 
		#print 'expected_negative_output:'
		#print expected_negative_output
		#print 

		# Check for equality between all found and expected values
		self.assertTrue(np.allclose(
			test_positive_output, expected_positive_output
		))
		self.assertTrue(np.allclose(
			test_negative_output, expected_negative_output
		))


	def test_Word2VecEmbedder(self):

		input_var = T.imatrix('input_var')

		embedder = Word2VecEmbedder(
			input_var,
			batch_size=len(self.TEST_INPUT),
			vocabulary_size=self.VOCAB_SIZE,
			num_embedding_dimensions=self.NUM_EMBEDDING_DIMENSIONS,
			word_embedding_init=self.QUERY_EMBEDDING,
			context_embedding_init=self.CONTEXT_EMBEDDING
		)

		query_embedding = embedder.query_embedding
		context_embedding = embedder.context_embedding
		dots = embedder.get_output()

		f = function([input_var], query_embedding)
		g = function([input_var], context_embedding)
		h = function([input_var], dots)
		
		# Calculate the embeddings and the output
		query_embeddings = f(self.TEST_INPUT)
		context_embeddings = g(self.TEST_INPUT)
		test_output = h(self.TEST_INPUT)

		# Calculate the expected embeddings and output
		expected_query_embeddings = np.repeat(
			self.QUERY_EMBEDDING, 3, axis=0
		)
		expected_context_embeddings = np.tile(
			self.CONTEXT_EMBEDDING, (3,1)
		)
		expected_output = usigma(np.dot(
			expected_query_embeddings, expected_context_embeddings.T
		)).diagonal()

		#print 'found query embeddings:'
		#print query_embeddings
		#print
		#print 'expected query embeddings:'
		#print expected_query_embeddings
		#print
		#print 'found context embeddings:'
		#print context_embeddings
		#print 
		#print 'expected context embeddings:'
		#print expected_context_embeddings
		#print 
		#print 'found output:'
		#print test_output
		#print
		#print 'expected output:'
		#print expected_output

		# Check for equality between all found and expected values
		self.assertTrue(np.allclose(
			query_embeddings, expected_query_embeddings
		))
		self.assertTrue(np.allclose(
			context_embeddings, expected_context_embeddings
		))
		self.assertTrue(np.allclose(test_output, expected_output))


if __name__=='__main__': 
	main()
	
