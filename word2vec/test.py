from unigram_dictionary import UnigramDictionary
from collections import Counter, defaultdict
from token_map import TokenMap, SILENT, ERROR, UNK
import time
from unittest import main, TestCase
from theano import tensor as T, function, shared
import numpy as np
from w2v import Word2VecEmbedder, word2vec
from noise_contrast import get_noise_contrastive_loss, noise_contrast
from dataset_reader import TokenChooser, DatasetReader, DataSetReaderIllegalStateException
from theano_minibatcher import (
	TheanoMinibatcher, NoiseContrastiveTheanoMinibatcher
)
from counter_sampler import CounterSampler
from lasagne.init import Normal
from lasagne.updates import nesterov_momentum
import re
import os
import shutil
from copy import deepcopy


def sigma(a):
	return 1/(1+np.exp(-a))
usigma = np.vectorize(sigma)


class TestUnigramDictionary(TestCase):
	'''
	Tests that UnigramDictionary properly represents the corpus
	statistics, and that the pruning function works as expected.
	'''

	TOKENS = ['apple', 'pear', 'banana', 'orange']

	# Make a toy corpus with specific token frequencies.
	FREQUENCIES = {
		'apple':4, 'banana':8, 'orange':6,
		'pineapple':3, 'grapefruit':9
	}
	CORPUS = list(Counter(FREQUENCIES).elements())

	def test_sort(self):
		np.random.seed(1)
		np.random.shuffle(self.CORPUS)
		unigram_dictionary = UnigramDictionary()
		unigram_dictionary.update(self.CORPUS)

		ordered_tokens = ['UNK'] + [
			token for token, count 
			in Counter(self.FREQUENCIES).most_common()
		]
		ordered_counts = [0] + [
			count for token, count 
			in Counter(self.FREQUENCIES).most_common()
		]

		self.assertNotEqual(
			unigram_dictionary.token_map.tokens,
			ordered_tokens
		)
		self.assertNotEqual(
			unigram_dictionary.counter_sampler.counts,
			ordered_counts
		)

		unigram_dictionary.sort()
		
		self.assertEqual(
			unigram_dictionary.token_map.tokens,
			ordered_tokens
		)
		self.assertEqual(
			unigram_dictionary.counter_sampler.counts,
			ordered_counts
		)

		


	def test_remove_compact(self):
		unigram_dictionary = UnigramDictionary()
		unigram_dictionary.update(self.CORPUS)
		unigram_dictionary.remove('banana')
		unigram_dictionary.remove('pineapple')
		unigram_dictionary.compact()
		
		adjusted_frequencies = deepcopy(self.FREQUENCIES)
		adjusted_frequencies['banana'] = 0
		adjusted_frequencies['pineapple'] = 0
		total = sum(adjusted_frequencies.values())
		reduced_tokens = [
			key for key, val in self.FREQUENCIES.iteritems()
			if val > 0
		]

		for token in reduced_tokens:
			expected_probability = (
				adjusted_frequencies[token] / float(total))
			idx = unigram_dictionary.get_id(token)
			self.assertEqual(
				unigram_dictionary.get_probability(idx), 
				expected_probability
			)

		# Attempting to remove tokens that don't exist is an error
		with self.assertRaises(ValueError):
			unigram_dictionary.remove('fake')

		# Attempting to remove the 'UNK' special token is an error
		with self.assertRaises(ValueError):
			unigram_dictionary.remove('UNK')


	def test_sampling(self):
		'''
		Test basic function of assigning counts, and then sampling from
		The distribution implied by those counts.
		'''
		unigram_dictionary = UnigramDictionary()
		unigram_dictionary.update(self.CORPUS)

		# Test asking for a single sample (where no shape tuple supplied)
		single_sample = unigram_dictionary.sample()
		self.assertTrue(type(single_sample) is np.int64)

		# Test asking for an array of samples (by passing a shape tuple)
		shape = (2,3,5)
		array_sample = unigram_dictionary.sample(shape)
		self.assertTrue(type(array_sample) is np.ndarray)
		self.assertTrue(array_sample.shape == shape)


	def test_counter_sampler_statistics(self):
		'''
		This tests that the UnigramDictionary really does produce results
		whose statistics match those requested by the counts vector
		'''
		# Seed numpy's random function to make the test reproducible
		np.random.seed(1)

		# Make a sampler with probabilities proportional to counts
		unigram_dictionary = UnigramDictionary()
		unigram_dictionary.update(self.CORPUS)

		# Draw one hundred thousand samples, then total up the fraction of
		# each outcome obseved
		counter = Counter(unigram_dictionary.sample((100000,)))

		# Make a list of the expected fractions by which each outcome
		# should be observed, in the limit of infinite sample
		total_in_expected = float(len(self.CORPUS))

		tolerance = 0.002
		for idx, found_freq in counter.iteritems():
			found_frac = found_freq / 100000.0
			token = unigram_dictionary.get_token(idx)
			expected_frac = self.FREQUENCIES[token] / total_in_expected
			self.assertTrue(abs(found_frac - expected_frac) < tolerance)


	def test_unigram_dictionary_token_map(self):

		unigram_dictionary = UnigramDictionary(on_unk=SILENT)

		for idx, fruit in enumerate(self.TOKENS):
			# Ensure that ids are assigned in an auto-incrementing way
			# starting from 1 (0 is reserved for the UNK token)
			self.assertEqual(unigram_dictionary.add(fruit), idx+1)

		for idx, fruit in enumerate(self.TOKENS):
			# Ensure that idxs are stable and retrievable with
			# UnigramDictionary.get_id()
			self.assertEqual(unigram_dictionary.get_id(fruit), idx+1)

			# Ensure that we can look up the token using the id
			self.assertEqual(unigram_dictionary.get_token(idx+1), fruit)

		# Ensure the unigram_dictionary knows its own length
		self.assertEqual(len(unigram_dictionary), len(self.TOKENS)+1)

		# Asking for ids of non-existent tokens returns the UNK token_id
		self.assertEqual(unigram_dictionary.get_id('no-exist'), 0)

		# Asking for the token at 0 returns 'UNK'
		self.assertEqual(unigram_dictionary.get_token(0), 'UNK')

		# Asking for token at non-existent idx raises IndexError
		with self.assertRaises(IndexError):
			unigram_dictionary.get_token(99)


	def test_raise_error_on_unk(self):
		'''
		If the unigram_dictionary is constructed passing
			on_unk=UnigramDictionary.ERROR
		then calling get_id() or get_ids() will throw a KeyError if one
		of the supplied tokens isn't in the unigram_dictionary.
		(Normally it would return 0, which is a token id reserved for
		'UNK' -- any unknown token).
		'''

		unigram_dictionary = UnigramDictionary(on_unk=ERROR)
		unigram_dictionary.update(self.TOKENS)

		with self.assertRaises(KeyError):
			unigram_dictionary.get_id('no-exist')

		with self.assertRaises(KeyError):
			unigram_dictionary.get_ids(['apple', 'no-exist'])



	def test_plural_functions(self):

		unigram_dictionary = UnigramDictionary(on_unk=SILENT)

		# In these assertions, we offset the expected list of ids by 1
		# because the 0th id in unigram_dictionary is reserved for the UNK
		# token

		# Ensure that update works
		ids = unigram_dictionary.update(self.TOKENS)
		self.assertEqual(ids, range(1, len(self.TOKENS)+1))

		# Ensure that get_ids works
		self.assertEqual(
			unigram_dictionary.get_ids(self.TOKENS),
			range(1, len(self.TOKENS)+1)
		)

		# Ensure that get_tokens works
		self.assertEqual(
			unigram_dictionary.get_tokens(range(1, len(self.TOKENS)+1)),
			self.TOKENS
		)

		# Asking for ids of non-existent tokens raises KeyError
		self.assertEqual(
			unigram_dictionary.get_ids(['apple', 'no-exist']),
			[self.TOKENS.index('apple')+1, 0]
		)

		# Asking for token at 0 returns the 'UNK' token
		self.assertEqual(
			unigram_dictionary.get_tokens([3,0]),
			[self.TOKENS[3-1], 'UNK']
		)

		# Asking for token at non-existent idx raises IndexError
		with self.assertRaises(IndexError):
			unigram_dictionary.get_tokens([1,99])


	def test_save_load(self):

		unigram_dictionary = UnigramDictionary(on_unk=SILENT)
		unigram_dictionary.update(self.CORPUS)
		unigram_dictionary.save('test-data/test-unigram-dictionary')

		unigram_dictionary_copy = UnigramDictionary(on_unk=SILENT)
		unigram_dictionary_copy.load('test-data/test-unigram-dictionary')

		# Test that the mapping from tokens to ids is unchanged
		for token in self.FREQUENCIES:
			self.assertEqual(
				unigram_dictionary.get_id(token),
				unigram_dictionary_copy.get_id(token)
			)

		# Test that the vocabulary size is as expected
		self.assertEqual(
			len(unigram_dictionary_copy),
			len(self.FREQUENCIES)+1
		)

		# Test that the counts for each token are correct
		for token, count in self.FREQUENCIES.items():
			token_id = unigram_dictionary.get_id(token)
			self.assertEqual(
				unigram_dictionary_copy.get_frequency(token_id),
				count
			)

		# Test that the number of tokens is as expected
		self.assertEqual(
			unigram_dictionary_copy.get_num_tokens(),
			sum(self.FREQUENCIES.values())
		)


	def test_pruning(self):


		# Make a unigram dictionary, and populate it with the corpus
		unigram_dictionary = UnigramDictionary()
		unigram_dictionary.update(self.CORPUS)

		# Ensure that the dictionary has correctly encoded the desired
		# information about the corpus.
		for token in unigram_dictionary.token_map.tokens:
			token_id = unigram_dictionary.get_id(token)
			freq = unigram_dictionary.get_frequency(token_id)
			if token == 'UNK':
				self.assertEqual(freq, 0)
			else:
				self.assertEqual(freq, self.FREQUENCIES[token])

		# Check that the dictionary knows the correct number of words.
		# Recall that this is one more than the number of unique words
		# in the corpus, because of the reserved 'UNK' word.
		num_tokens = len(self.FREQUENCIES)+1
		self.assertEqual(len(unigram_dictionary), num_tokens)

		# Prune the dictionary!
		unigram_dictionary.prune(min_frequency=6)

		# Check that two elements were dropped from the dictionary
		# ('apple' and 'pineapple').
		num_tokens -= 2
		self.assertEqual(len(unigram_dictionary), num_tokens)

		# Check that the frequences are as expected.  Counts for
		# apple and pineapple should have been attributed to UNK
		unk_freq = (
			self.FREQUENCIES['apple'] + self.FREQUENCIES['pineapple']
		)
		for token in unigram_dictionary.token_map.tokens:

			# We should not see apple or pineapple
			self.assertTrue(token not in ('apple', 'pineapple'))

			token_id = unigram_dictionary.get_id(token)
			freq = unigram_dictionary.get_frequency(token_id)
			if token == 'UNK':
				self.assertEqual(freq, unk_freq)
			else:
				self.assertEqual(freq, self.FREQUENCIES[token])


class TestTokenChooser(TestCase):
	'''
	Given a list of tokens (usually representing a sentence), and a
	particular query token, the job of the TokenChooser is to sample
	from the nearby tokens, within a window of +/- K tokens.
	It does not yield the sampled token itself, but rather, its index.
	A complicating issue is the fact that the query token might be
	near the beginning or end of the token list (sentence), which needs
	to be accounted for so that the sampled index doesn't produce an
	IndexError
	'''

	def test_mid_sentence(self):
		'''
		Test token chooser's performance given that the query token
		is far from the edges of the context
		'''

		# Seed randomness for reproducibility
		np.random.seed(1)

		# Set up some testing data.  We'll sample from this sentence
		# using the word "sufficiently" as the query word
		sentence = (
			'this is a test sentence that is sufficiently long to '
			'enable testing various edge cases'
		).split()
		query_idx = sentence.index('sufficiently')
		context_length = len(sentence)

		# Sample from +/- 5 words
		K = 5
		# Weight the probabilities like this (higher probability of
		# sampling near the query word itself).
		counts = [1,2,3,4,5,5,4,3,2,1]

		# Make the chooser
		chooser = TokenChooser(K, counts)

		# Sample many times so we can test the sample's statistics
		num_samples = 100000
		found_counts = Counter([
			chooser.choose_token(query_idx, context_length)
			for s in range(num_samples)
		])

		# Convert counts into frequencies
		found_frequencies = dict([
			(idx, c / float(num_samples))
			for idx, c in found_counts.items()
		])

		# Assemble the expected frequencies.  First, the acceptable
		# outcomes are tokens that are within +/- 5.  So the acceptable
		# relative indices are:
		expected_relative_idxs = range(-5, 0) + range(1, 6)

		# And so the expected absolute indices are:
		expected_idxs = [
			rel + query_idx
			for rel in expected_relative_idxs
		]

		# Calculate the expected frequency that each index should appear
		# in the sample
		total = float(sum(counts))
		expected_frequencies = dict(
			[(idx, c / total) for idx, c in zip(expected_idxs, counts)]
		)

		# First, make sure that the sample indices are the ones expected
		self.assertEqual(
			set(found_frequencies.keys()),
			set(expected_frequencies.keys())
		)

		# Then check that the indices in the sample arise with
		# approximately the frequencies expected
		tolerance = 0.002
		for idx in expected_frequencies:
			diff = abs(expected_frequencies[idx] - found_frequencies[idx])
			self.assertTrue(diff < tolerance)


	def test_near_beginning(self):
		'''
		Test token chooser's performance given that the query token
		is close to the beginning of the context
		'''

		# Seed randomness for reproducibility
		np.random.seed(1)

		# Set up some testing data.  We'll sample from this sentence
		# using the word "a" as the query word
		sentence = (
			'this is a test sentence that is sufficiently long to '
			'enable testing various edge cases'
		).split()
		query_idx = sentence.index('a')
		context_length = len(sentence)

		# Sample from +/- 5 words
		K = 5
		# Weight the probabilities like this (higher probability of
		# sampling near the query word itself).
		counts = [1,2,3,4,5,5,4,3,2,1]

		# Make the chooser
		chooser = TokenChooser(K, counts)

		# Sample many times so we can test the sample's statistics
		num_samples = 100000
		found_counts = Counter([
			chooser.choose_token(query_idx, context_length)
			for s in range(num_samples)
		])

		# Convert counts into frequencies
		found_frequencies = dict([
			(idx, c / float(num_samples))
			for idx, c in found_counts.items()
		])

		# Assemble the expected frequencies.  First, the acceptable
		# outcomes are tokens that are within +/- 5.  But because the
		# query token is near the start of the context, there are actually
		# only two tokens available before it
		expected_relative_idxs = [-2, -1] + range(1, 6)

		# And so the expected absolute indices are:
		expected_idxs = [
			rel + query_idx
			for rel in expected_relative_idxs
		]

		# Calculate the expected frequency that each index should appear
		# in the sample
		relative_frequencies = [4,5,5,4,3,2,1]
		total = float(sum(relative_frequencies))
		expected_frequencies = dict([
			(idx, c / total)
			for idx, c in zip(expected_idxs, relative_frequencies)
		])

		# First, make sure that the sample indices are the ones expected
		self.assertEqual(
			set(found_frequencies.keys()),
			set(expected_frequencies.keys())
		)

		# Then check that the indices in the sample arise with
		# approximately the frequencies expected
		tolerance = 0.003
		for idx in expected_frequencies:
			diff = abs(expected_frequencies[idx] - found_frequencies[idx])
			self.assertTrue(diff < tolerance)


	def test_near_end(self):
		'''
		Test token chooser's performance given that the query token
		is close to the end of the context
		'''

		# Seed randomness for reproducibility
		np.random.seed(1)

		# Set up some testing data.  We'll sample from this sentence
		# using the word "cases" as the query word
		sentence = (
			'this is a test sentence that is sufficiently long to '
			'enable testing various edge cases'
		).split()
		query_idx = sentence.index('cases')
		context_length = len(sentence)

		# Sample from +/- 5 words
		K = 5
		# Weight the probabilities like this (higher probability of
		# sampling near the query word itself).
		counts = [1,2,3,4,5,5,4,3,2,1]

		# Make the chooser
		chooser = TokenChooser(K, counts)

		# Sample many times so we can test the sample's statistics
		num_samples = 100000
		found_counts = Counter([
			chooser.choose_token(query_idx, context_length)
			for s in range(num_samples)
		])

		# Convert counts into frequencies
		found_frequencies = dict([
			(idx, c / float(num_samples))
			for idx, c in found_counts.items()
		])

		# Assemble the expected frequencies.  First, the acceptable
		# outcomes are tokens that are within +/- 5.  But because the
		# query token is near the start of the context, there are actually
		# only two tokens available before it
		expected_relative_idxs = range(-5, 0)

		# And so the expected absolute indices are:
		expected_idxs = [
			rel + query_idx
			for rel in expected_relative_idxs
		]

		# Calculate the expected frequency that each index should appear
		# in the sample
		relative_frequencies = [1,2,3,4,5]
		total = float(sum(relative_frequencies))
		expected_frequencies = dict([
			(idx, c / total)
			for idx, c in zip(expected_idxs, relative_frequencies)
		])

		# First, make sure that the sample indices are the ones expected
		self.assertEqual(
			set(found_frequencies.keys()),
			set(expected_frequencies.keys())
		)

		# Then check that the indices in the sample arise with
		# approximately the frequencies expected
		tolerance = 0.003
		for idx in expected_frequencies:
			diff = abs(expected_frequencies[idx] - found_frequencies[idx])
			self.assertTrue(diff < tolerance)


	def test_short_context(self):
		'''
		Test token chooser's performance given that context is short
		'''

		# Seed randomness for reproducibility
		np.random.seed(1)

		# Set up some testing data.  We'll sample from this sentence
		# using the word "This" as the query word
		sentence = 'This is short'.split()
		query_idx = sentence.index('This')
		context_length = len(sentence)

		# Sample from +/- 5 words
		K = 5
		# Weight the probabilities like this (higher probability of
		# sampling near the query word itself).
		counts = [1,2,3,4,5,5,4,3,2,1]

		# Make the chooser
		chooser = TokenChooser(K, counts)

		# Sample many times so we can test the sample's statistics
		num_samples = 100000
		found_counts = Counter([
			chooser.choose_token(query_idx, context_length)
			for s in range(num_samples)
		])

		# Convert counts into frequencies
		found_frequencies = dict([
			(idx, c / float(num_samples))
			for idx, c in found_counts.items()
		])

		# Assemble the expected frequencies.  First, the acceptable
		# outcomes are tokens that are within +/- 5.  But because the
		# query token is near the start of the context, there are actually
		# only two tokens available before it
		expected_relative_idxs = [1,2]

		# And so the expected absolute indices are:
		expected_idxs = [
			rel + query_idx
			for rel in expected_relative_idxs
		]

		# Calculate the expected frequency that each index should appear
		# in the sample
		relative_frequencies = [5,4]
		total = float(sum(relative_frequencies))
		expected_frequencies = dict([
			(idx, c / total)
			for idx, c in zip(expected_idxs, relative_frequencies)
		])

		# First, make sure that the sample indices are the ones expected
		self.assertEqual(
			set(found_frequencies.keys()),
			set(expected_frequencies.keys())
		)

		# Then check that the indices in the sample arise with
		# approximately the frequencies expected
		tolerance = 0.003
		for idx in expected_frequencies:
			diff = abs(expected_frequencies[idx] - found_frequencies[idx])
			self.assertTrue(diff < tolerance)



class TestCounterSampler(TestCase):

	def test_sampling(self):
		'''
		Test basic function of assigning counts, and then sampling from
		The distribution implied by those counts.
		'''
		counts = range(1,6)
		counter_sampler = CounterSampler()
		counter_sampler.update(counts)

		# Test asking for a single sample (where no shape tuple supplied)
		single_sample = counter_sampler.sample()
		self.assertTrue(type(single_sample) is np.int64)

		# Test asking for an array of samples (by passing a shape tuple)
		shape = (2,3,5)
		array_sample = counter_sampler.sample(shape)
		self.assertTrue(type(array_sample) is np.ndarray)
		self.assertTrue(array_sample.shape == shape)


	def test_add_function(self):
		'''
		Make sure that the add function is working correctly.
		CounterSampler stores counts as list, wherein the value at
		position i of the list encodes the number of counts seen for
		outcome i.

		Counts are added by passing the outcome's index into
		CounterSampler.add()
		which leads to position i of the counts list to be incremented.
		If position i doesn't exist, it is created.  If the counts list
		had only j elements before, and a count is added for position
		i, with i much greater than j, then many elements are created
		between i and j, and are provisionally initialized with zero
		counts.

		Ensure that is done properly
		'''

		counter_sampler = CounterSampler()
		self.assertEqual(counter_sampler.counts, [])

		outcome_to_add = 6
		counter_sampler.add(outcome_to_add)
		expected_counts = [0]*(outcome_to_add) + [1]
		self.assertEqual(counter_sampler.counts, expected_counts)

		# Now ensure the underlying sampler can tolerate a counts list
		# containing zeros, and that the sampling statistics is as
		# expected.  We expect that the only outcome that should turn up
		# is outcome 6, since it has all the probability mass.  Check that.
		counter = Counter(counter_sampler.sample((100000,))) # should be
															 # all 6's
		total = float(sum(counter.values()))
		found_normalized = [
			counter[i] / total for i in range(outcome_to_add+1)
		]

		# Make an list of the expected fractions by which each outcome
		# should be observed, in the limit of infinite sample
		expected_normalized = expected_counts

		# Check if each outcome was observed with a fraction that is within
		# 0.005 of the expected fraction
		self.assertEqual(found_normalized, expected_normalized)


	def test_counter_sampler_statistics(self):
		'''
		This tests that the sampler really does produce results whose
		statistics match those requested by the counts vector
		'''
		# Seed numpy's random function to make the test reproducible
		np.random.seed(1)

		# Make a sampler with probabilities proportional to counts
		counts = range(1,6)
		counter_sampler = CounterSampler()
		for outcome, count in enumerate(counts):
			counter_sampler.update([outcome]*count)

		# Draw one hundred thousand samples, then total up the fraction of
		# each outcome obseved
		counter = Counter(counter_sampler.sample((100000,)))
		total = float(sum(counter.values()))
		found_normalized = [
			counter[i] / total for i in range(len(counts))
		]

		# Make an list of the expected fractions by which each outcome
		# should be observed, in the limit of infinite sample
		total_in_expected = float(sum(counts))
		expected_normalized = [
			c / total_in_expected for c in counts
		]

		# Check if each outcome was observed with a fraction that is within
		# 0.005 of the expected fraction
		close = [
			abs(f - e) < 0.005
			for f,e in zip(found_normalized, expected_normalized)
		]
		self.assertTrue(all(close))


	def test_save_load(self):

		fname = 'test-data/test-counter-sampler/test-counter-sampler.gz'

		# Make a sampler with probabilities proportional to counts
		counts = range(1,6)
		counter_sampler = CounterSampler()
		for outcome, count in enumerate(counts):
			counter_sampler.update([outcome]*count)

		counter_sampler.save(fname)

		new_counter_sampler = CounterSampler()
		new_counter_sampler.load(fname)
		self.assertEqual(new_counter_sampler.counts, counts)


class TestTokenMap(TestCase):

	TOKENS = ['apple', 'pear', 'banana', 'orange']

	def test_token_map(self):

		token_map = TokenMap(on_unk=SILENT)

		for idx, fruit in enumerate(self.TOKENS):
			# Ensure that ids are assigned in an auto-incrementing way
			# starting from 1 (0 is reserved for the UNK token)
			self.assertEqual(token_map.add(fruit), idx+1)

		for idx, fruit in enumerate(self.TOKENS):
			# Ensure that idxs are stable and retrievable with
			# TokenMap.get_id()
			self.assertEqual(token_map.get_id(fruit), idx+1)

			# Ensure that we can look up the token using the id
			self.assertEqual(token_map.get_token(idx+1), fruit)

		# Ensure the token_map knows its own length
		self.assertEqual(len(token_map), len(self.TOKENS)+1)

		# Asking for ids of non-existent tokens returns the UNK token_id
		self.assertEqual(token_map.get_id('no-exist'), 0)

		# Asking for the token at 0 returns 'UNK'
		self.assertEqual(token_map.get_token(0), 'UNK')

		# Asking for token at non-existent idx raises IndexError
		with self.assertRaises(IndexError):
			token_map.get_token(99)


	def test_raise_error_on_unk(self):
		'''
		If the token_map is constructed passing
			on_unk=TokenMap.ERROR
		then calling get_id() or get_ids() will throw a KeyError if one
		of the supplied tokens isn't in the token_map.  (Normally it
		would return 0, which is a token id reserved for 'UNK' -- any
		unknown token).
		'''

		token_map = TokenMap(on_unk=ERROR)
		token_map.update(self.TOKENS)

		with self.assertRaises(KeyError):
			token_map.get_id('no-exist')

		with self.assertRaises(KeyError):
			token_map.get_ids(['apple', 'no-exist'])


	def test_token_map_plural_functions(self):

		token_map = TokenMap(on_unk=SILENT)

		# In these assertions, we offset the expected list of ids by
		# 1 because the 0th id in token_map is reserved for the UNK
		# token

		# Ensure that update works
		ids = token_map.update(self.TOKENS)
		self.assertEqual(ids, range(1, len(self.TOKENS)+1))

		# Ensure that get_ids works
		self.assertEqual(
			token_map.get_ids(self.TOKENS),
			range(1, len(self.TOKENS)+1)
		)

		# Ensure that get_tokens works
		self.assertEqual(
			token_map.get_tokens(range(1, len(self.TOKENS)+1)),
			self.TOKENS
		)

		# Asking for ids of non-existent tokens raises KeyError
		self.assertEqual(
			token_map.get_ids(['apple', 'no-exist']),
			[self.TOKENS.index('apple')+1, 0]
		)

		# Asking for token at 0 returns the 'UNK' token
		self.assertEqual(
			token_map.get_tokens([3,0]),
			[self.TOKENS[3-1], 'UNK']
		)

		# Asking for token at non-existent idx raises IndexError
		with self.assertRaises(IndexError):
			token_map.get_tokens([1,99])


	def test_save_load(self):
		token_map = TokenMap(on_unk=SILENT)
		token_map.update(self.TOKENS)
		token_map.save('test-data/test-token-map/test-token-map.gz')

		token_map_copy = TokenMap(on_unk=SILENT)
		token_map_copy.load(
			'test-data/test-token-map/test-token-map.gz'
		)
		self.assertEqual(
			token_map_copy.get_ids(self.TOKENS),
			range(1, len(self.TOKENS)+1)
		)
		self.assertEqual(len(token_map_copy), len(self.TOKENS)+1)



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

		self.assertAlmostEqual(test_loss, expected_loss)


class TestDataReader(TestCase):

	#TODO: add a test for saving and loading the dictionary and for 
	def setUp(self):

		# Define some parameters to be used in construction
		# Minibatcher
		self.files = [
			'test-data/test-corpus/003.tsv',
			'test-data/test-corpus/004.tsv'
		]
		self.batch_size = 5
		self.macrobatch_size = 5
		self.noise_ratio = 15
		self.num_example_generators = 3
		self.t = 0.03

		self.dataset_reader_with_discard = DatasetReader(
			files=self.files,
			noise_ratio = self.noise_ratio,
			t=self.t,
			macrobatch_size=self.macrobatch_size,
			num_processes=3,
			verbose=False
		)

		self.dataset_reader_no_discard = DatasetReader(
			files=self.files,
			macrobatch_size=self.macrobatch_size,
			noise_ratio = self.noise_ratio,
			t=1.0,
			num_processes=3,
			verbose=False
		)


	def test_prune(self):
		save_dir = 'test-data/test-dataset-reader'
		if os.path.exists(save_dir):
			shutil.rmtree(save_dir)

		files = [
			'test-data/test-corpus/003.tsv',
			'test-data/test-corpus/004.tsv'
		]

		# First make a dataset reader with no min frequency
		reader = DatasetReader(
			files=files,
			min_frequency=0,
			verbose=False
		)
		reader.prepare(save_dir=save_dir)

		# There should be 303 tokens in the dictionary
		self.assertEqual(reader.get_vocab_size(), 303)

		# Now make a dataset reader with a min frequency of 10
		reader = DatasetReader(
			files=files,
			min_frequency=10,
			verbose=False
		)
		reader.prepare()

		# There should be only 7 tokens in dictionary (others got pruned)
		self.assertEqual(reader.get_vocab_size(), 7)

		# Now try making a dataset reader and loading from file,
		# again enforcing the in frequency
		reader = DatasetReader(
			files=files,
			min_frequency=10,
			load_dictionary_dir=save_dir,
			verbose=False
		)
		self.assertEqual(reader.get_vocab_size(), 7)

		# Now try making a dataset reader and loading the dictionary
		# manually
		reader = DatasetReader(
			files=files,
			min_frequency=10,
			verbose=False
		)
		reader.load_dictionary(save_dir)
		self.assertEqual(reader.get_vocab_size(), 7)

		# Now try passing the dictionary into the DatasetReader
		dictionary = UnigramDictionary()
		dictionary.load(os.path.join(save_dir, 'dictionary'))
		reader = DatasetReader(
			files=files,
			min_frequency=10,
			unigram_dictionary=dictionary,
			verbose=False
		)
		self.assertEqual(reader.get_vocab_size(), 7)



	def test_illegal_state_exception(self):
		'''
		Calling generate_dataset_parallel() on a DatasetReader before 
		calling prepare() should raise DataSetReaderIllegalStateException.
		'''

		reader = self.dataset_reader_with_discard
		with self.assertRaises(DataSetReaderIllegalStateException):
			iterator = reader.generate_dataset_parallel()
			iterator.next()

		with self.assertRaises(DataSetReaderIllegalStateException):
			iterator = reader.generate_dataset_serial()
			iterator.next()


	def test_token_discarding(self):
		'''
		Make sure that token discarding is occurring as expected.  We build
		the test case around the token "the", because its the most common 
		token, and so is most reliably discarded.  We compare the number of
		signal examples whose query word is "the" to the number of 
		occurrences of "the" in the test corpus, and check that this 
		fraction is close to that expected based on the prescribed 
		probability of discarding.
		'''
		# Ensure reproducibility in this stochastic test
		np.random.seed(1)
		reader = self.dataset_reader_with_discard
		reader.prepare()

		# Count the number of times "the" appears
		num_the_tokens = 0
		the_pattern = re.compile(r'\bthe\b')
		for filename in self.files:
			text = open(filename).read()
			the_matches = the_pattern.findall(text)
			num_the_tokens += len(the_matches)

		# Repeatedly generate the dataset, with discarding, and keep track 
		# of how many times "the" is included as a query word
		num_replicates = 5
		signal_query_frequencies = []
		for i in range(num_replicates):
			signal_query_freq, noise_query_freq = (
				self.get_the_query_frequency())
			signal_query_frequencies.append(signal_query_freq)
			# The ratio of noise and signal examples should not be affected
			# by discarding.
			self.assertEqual(
				signal_query_freq*self.noise_ratio, noise_query_freq)

		# Take the average frequency with which "the" arose accross 
		# replicates.
		signal_query_frequency = np.mean(signal_query_frequencies)

		# We expect "the" to arise in signal queries less often than it 
		# actually arises in the text, based on the probability of 
		# discarding
		tolerance = 0.05
		the_token = reader.unigram_dictionary.get_id('the')
		the_frequency = reader.unigram_dictionary.get_probability(
			the_token)
		expected_keep_ratio = np.sqrt(self.t/the_frequency)
		actual_keep_ratio = signal_query_frequency / float(num_the_tokens)
		self.assertTrue(
			abs(actual_keep_ratio - expected_keep_ratio) < tolerance
		)


	def get_the_query_frequency(self):
		# Generate the dataset, and see how many times "the" appears as a 
		# query word.
		reader = self.dataset_reader_with_discard
		dataset = reader.generate_dataset_parallel()
		seen_noise_queries = Counter()
		num_the_signals = 0
		num_the_noises = 0
		the_token = reader.unigram_dictionary.get_id('the')
		dataset_iterator = self.iterate_dataset(dataset)
		for signal_examples, noise_examples in dataset_iterator:

			# Count number of times "the" occurs as query in 
			# signal_examples
			signal_queries = signal_examples[:,0]
			num_the_signals += sum([
				t == the_token for t in signal_queries
			])

			# Count number of times "the" occurs as query in noise_examples
			noise_queries = noise_examples[:,0]
			num_the_noises += sum([t == the_token for t in noise_queries])

		return num_the_signals, num_the_noises


	def test_dataset_composition(self):
		'''
		Make sure that the minibatches are the correct size, that
		signal query- and contexts-words are always within 5 tokens of
		one another and come from the same sentence.
		'''
		# Ensure reproducibility in this stochastic test
		np.random.seed(1)

		reader = self.dataset_reader_no_discard
		reader.prepare()

		# Iterate through the corpus, noting what tokens arise within
		# one another's contexts.  Build a lookup table, indicating the set
		# of "legal pairs" -- tokens that arose in one another's context.
		legal_pairs = defaultdict(set)
		# We'll also keep track of the query words in the signal examples
		# To make sure that noise examples are also made for them
		expected_noise_queries = Counter()
		d = reader.unigram_dictionary
		for filename in self.files:
			for tokens in reader.parse(filename):
				token_ids = d.get_ids(tokens)
				for i, token_id in enumerate(token_ids):
					low = max(0, i-5)
					legal_context = token_ids[low:i] + token_ids[i+1:i+6]
					legal_pairs[token_id].update(legal_context)
					# Every time a token appears we expect noise_ratio 
					# noise examples for it
					expected_noise_queries[token_id] += self.noise_ratio

		# finally, and the pair (UNK, UNK), which is used to pad data
		legal_pairs[UNK] = set([UNK])

		reader.prepare()
		dataset = reader.generate_dataset_parallel()
		seen_noise_queries = Counter()
		dataset_iterator = self.iterate_dataset(dataset)
		for signal_examples, noise_examples in dataset_iterator:

			# Keep track of how many times each token appears as the query 
			# in a noise example
			noise_queries = noise_examples[:,0]
			seen_noise_queries.update(noise_queries)

			# Ensure that all of the signal examples are actually valid
			# samples from the corpus
			for query, context in signal_examples:
				self.assertTrue(context in legal_pairs[query])

		# Ensure that we got the expected number of appearances of tokens 
		# in noise examples.  But since we don't care about the number of 
		# times that UNK appears (since it is used for padding), we remove 
		# it from observed counts first
		del seen_noise_queries[UNK]
		keys = set(
			seen_noise_queries.keys() + expected_noise_queries.keys())
		for key in keys:
			self.assertEqual(
				seen_noise_queries[key], expected_noise_queries[key])


	def iterate_dataset(self, dataset):
		'''
		Given a dataset, separate out successive signal and noise examples,
		and iterate through them
		'''

		batch_size = self.batch_size
		noise_batch_size = self.batch_size * self.noise_ratio

		for signal_macrobatch, noise_macrobatch in dataset:
			num_batches = len(signal_macrobatch) / self.batch_size
			for pointer in range(num_batches):

				signal_start = pointer * batch_size
				signal_examples = signal_macrobatch[
					signal_start : signal_start + batch_size
				]
				noise_start = pointer * noise_batch_size
				noise_examples = noise_macrobatch[
					noise_start : noise_start + noise_batch_size
				]

				yield signal_examples, noise_examples


	def test_prepare(self):
		'''
		Check that DatasetReader.prepare() properly makes a
		UnigramDictionary that reflects the corpus.
		'''
		reader = self.dataset_reader_with_discard
		reader.prepare()
		d = reader.unigram_dictionary

		# Make sure that all of the expected tokens are found in the
		# unigram_dictionary, and that their frequency in the is correct.
		tokens = []
		for filename in self.files:
			for add_tokens in reader.parse(filename):
				tokens.extend(add_tokens)

		counts = Counter(tokens)
		for token in tokens:
			token_id = d.get_id(token)
			count = d.get_frequency(token_id)
			self.assertEqual(count, counts[token])


	def test_noise_uniqueness(self):
		'''
		A bug was discovered whereby, when batches are yielded via the
		asynchronous method (i.e. via the iterator returned by
		Minibatcher.get_async_batch_iterator()), different example
		generating processes will generate the same sequence of noise
		contexts, because each example generating process inherits the
		same numpy random number generating state.  This test exposes the
		bug, so that it could be fixed.
		'''

		# Ensure reproducibility in this stochastic test
		np.random.seed(1)

		reader = self.dataset_reader_no_discard
		reader.prepare()
		dataset = reader.generate_dataset_parallel()

		# Iterate through all examples in the dataset.  Look at each of
		# the noise examples generated.  Each query word should have
		# noise_ratio number of noise contexts generated.  No two
		# query words should get the same sequence of noise sequences,
		# which is what we want to test.  Collect the sequence of noise
		# contexts for each query word, and check they are unique
		noise_sequences = defaultdict(list)
		last_query = None
		dataset_iterator = self.iterate_dataset(dataset)
		for signal_examples, noise_examples in dataset_iterator:
			for row in noise_examples:
				query, context = row
				# We don't care about noise generated for UNK queries: they
				# will certainly repeat because UNK is used for padding
				if query == UNK:
					continue
				noise_sequences[query].append(context)


		# Now iterate through all of the accumulated noise sequences, and
		# ensure that we don't see any duplicates
		seen_noise_sequences = set()
		for key in noise_sequences:

			# The full sequence involves noise sequences for multiple 
			# instances of the query.  Divide it into chunks, where each is
			# the noise sequence for an individual instance
			full_sequence = noise_sequences[key]
			instance_sequences = [
				tuple(full_sequence[i:i+self.noise_ratio])
				for i in xrange(0, len(full_sequence), self.noise_ratio)
			]

			# Check if any of these sequences happened already, and add it 
			# to the list of seen noise sequences
			for sequence in instance_sequences:
				self.assertFalse(sequence in seen_noise_sequences)
				seen_noise_sequences.add(sequence)


#	TODO: saving and loading has been disabled -- it doesn't seem to 
# 	make sense under generator-based dataset iteration
#
#	def test_save_load_data_serial(self):
#		self.save_load_data(True)
#
#
#	def test_save_load_data_parallel(self):
#		self.save_load_data(False)
#
#
#	def save_load_data(self, use_serial):
#		'''
#		Test that save and load actually write and read the 
#		Word2VecEmbedder's parameters, by testing that a saved and then 
#		then loaded model yields the same model as the original.
#		'''
#
#		# Clear old testing data, and make sure dir for testing data exists
#		directory = 'test-data/test-dataset-reader'
#		if os.path.exists(directory):
#			shutil.rmtree(directory)
#		os.mkdir(directory)
#
#		# Make a dataset reader, and use two files as the dataset.  Use no
#		# token discarding.
#		reader = DatasetReader(
#			files=[
#				'test-data/test-corpus/numbers-med1.txt',
#				'test-data/test-corpus/numbers-med2.txt'
#			],
#			t=1, num_processes = 2,
#			verbose=False
#		)
#		reader.prepare()
#
#		# Generate the dataset, and pass in `directory` as the location to 
#		# save the generated dataset
#		if use_serial:
#			reader.generate_dataset_serial(directory)
#		else:
#			reader.generate_dataset_parallel(directory)
#
#		# Reload the saved dataset from file
#		new_reader = DatasetReader()
#		new_reader.load_data(directory)
#
#		# Ensure the generated dataset, and the one relaoded from file 
#		# are equal.
#		self.assertTrue(
#			np.array_equal(new_reader.examples, reader.examples))


class TestMinibatcher(TestCase):

	def setUp(self):

		# Define some parameters to be used in construction
		# Minibatcher
		self.files = [
			'test-data/test-corpus/003.tsv',
			'test-data/test-corpus/004.tsv'
		]
		self.batch_size = 5
		self.macrobatch_size = 20
		self.noise_ratio = 15
		self.num_example_generators = 3
		self.t = 0.03

		self.dataset_reader_with_discard = DatasetReader(
			files=self.files,
			noise_ratio = self.noise_ratio,
			t=self.t,
			num_processes=3,
			macrobatch_size=self.macrobatch_size,
			verbose=False
		)

		self.dataset_reader_no_discard = DatasetReader(
			files=self.files,
			noise_ratio = self.noise_ratio,
			t=1.0,
			num_processes=3,
			macrobatch_size=self.macrobatch_size,
			verbose=False
		)


	def test_dataset_composition(self):
		'''
		Make sure that the minibatches are the correct size, that
		signal query- and contexts-words are always within 5 tokens of
		one another and come from the same sentence.
		'''
		# Ensure reproducibility in this stochastic test
		np.random.seed(1)

		reader = self.dataset_reader_no_discard
		reader.prepare()

		# Iterate through the corpus, noting what tokens arise within
		# one another's contexts.  Build a lookup table, indicating the set
		# of "legal pairs" -- tokens that arose in one another's context.
		legal_pairs = defaultdict(set)
		# We'll also keep track of the query words in the signal examples
		# To make sure that noise examples are also made for them
		expected_noise_queries = Counter()
		d = reader.unigram_dictionary
		for filename in self.files:
			for tokens in reader.parse(filename):
				token_ids = d.get_ids(tokens)
				for i, token_id in enumerate(token_ids):
					low = max(0, i-5)
					legal_context = token_ids[low:i] + token_ids[i+1:i+6]
					legal_pairs[token_id].update(legal_context)
					# Every time a token appears we expect noise_ratio 
					# noise examples for it
					expected_noise_queries[token_id] += self.noise_ratio

		# finally, and the pair (UNK, UNK), which is used to pad data
		legal_pairs[UNK] = set([UNK])

		# Make a minibatcher
 		minibatcher = NoiseContrastiveTheanoMinibatcher(
			batch_size=self.batch_size, 
			noise_ratio=self.noise_ratio, 
			dtype='int32',
			num_dims=2
		)
		symbolic_batch = minibatcher.get_batch()
		updates = minibatcher.get_updates()



		# Mock a theano training function.  The function simply returns the
		# current minibatch, so we can check that the minibatches have the
		# expected composition
		f = function([], symbolic_batch, updates=updates)

		seen_noise_queries = Counter()
		for macrobatch in reader.generate_dataset_parallel():
			signal_macrobatch, noise_macrobatch = macrobatch

			# Load the dataset
			num_batches = minibatcher.load_dataset(
				signal_macrobatch, noise_macrobatch
			)

			# Iterate through the minibatches, to check that they have 
			# expected composition
			for i in range(num_batches):

				# Get the batch, and split it into the noise and signal 
				# parts
				batch = f()
				signal_examples = batch[0:self.batch_size, ]
				noise_examples = batch[
					self.batch_size:self.batch_size*(self.noise_ratio+1), ]

				# Keep track of how many times each token appears as the 
				# query in a noise example
				noise_queries = noise_examples[:,0]
				seen_noise_queries.update(noise_queries)

				# Ensure that all of the signal examples are actually valid
				# samples from the corpus
				for query, context in signal_examples:
					self.assertTrue(context in legal_pairs[query])

		# Ensure that we got the expected number of appearances of tokens 
		# in noise examples.  But since we don't care about the number of 
		# times that UNK appears (since it is used for padding), we remove 
		# it from observed counts first
		del seen_noise_queries[UNK]
		keys = set(
			seen_noise_queries.keys() + expected_noise_queries.keys()
		)
		for key in keys:
			self.assertEqual(
				seen_noise_queries[key], expected_noise_queries[key])


	def test_symbolic_minibatches(self):
		'''
		Test that the symbolic minibatching mechanism yields the expected
		batches when used in a compiled theno function.
		'''
		# Ensure reproducibility in this stochastic test
		np.random.seed(1)

		# Make a TheanoMinibatcher; get the symbolic minibatch and updates
		batch_size = 5
		num_batches = 10
		num_examples = batch_size * num_batches
		dtype = 'int32'
		num_dims = 2
		batcher = TheanoMinibatcher(batch_size, dtype, num_dims)
		symbolic_batch = batcher.get_batch()
		updates = batcher.get_updates()

		# Mock a training function.  It just returns the current batch.
		f = function([], symbolic_batch, updates=updates)

		# Load a mock dataset
		dataset = np.array([[i,i] for i in range(num_examples)], dtype=dtype)
		actual_num_batches = batcher.load_dataset(dataset)

		# Check that we have the expected number of batches
		self.assertEqual(actual_num_batches, num_batches)

		for batch_num in range(actual_num_batches):
			batch = f()
			start, stop = batch_size*batch_num, batch_size*(batch_num+1)
			expected_batch = np.array([[i,i] for i in range(start, stop)], dtype=dtype)
			self.assertTrue(np.array_equal(batch, expected_batch))


	def test_data_container_initialization(self):

		# Make a minibatcher so we can test its _initialize_data_container method
		batcher = TheanoMinibatcher()

		# Test the method at various parameter values
		float32_1 = batcher._initialize_data_container(1,'float32')
		uint32_2 = batcher._initialize_data_container(2,'uint32')
		float64_3 = batcher._initialize_data_container(3,'float64')

		# Mock the expected results
		expected_float32_1 = np.array([],dtype='float32')
		expected_uint32_2 = np.array([[]],dtype='uint32')
		expected_float64_3 = np.array([[[]]],dtype='float64')

		# Ensure we got the expected results
		self.assertTrue(np.array_equal(float32_1, expected_float32_1))
		self.assertTrue(np.array_equal(uint32_2, expected_uint32_2))
		self.assertTrue(np.array_equal(float64_3, expected_float64_3))

		# The number of dimensions in a loaded dataset needs to match the number
		# of dimensions given to the TheanoMinibatcher constructor
		batcher = TheanoMinibatcher(num_dims=3, dtype='float32')
		# This works
		batcher.load_dataset(np.array([[[1.0,2.0]]], dtype='float32'))
		# But not this
		with self.assertRaises(TypeError):
			batcher.load_dataset(np.array([[1.0,2.0]], dtype='float32'))

		# Trying to initialize a 0-dimensional dataset should raise a ValueError
		with self.assertRaises(ValueError):
			int32_0 = batcher._initialize_data_container(0,'int32')


	def test_symbolic_minibatches_more_dims(self):
		'''
		Test that the symbolic minibatching mechanism yields the expected
		batches when used in a compiled theno function, and specifically test
		using a non-default number of dimensions in the dataset.
		'''
		# Ensure reproducibility in this stochastic test
		np.random.seed(1)

		# Make a TheanoMinibatcher; get the symbolic minibatch and updates
		batch_size = 5
		num_batches = 10
		num_examples = batch_size * num_batches
		dtype = 'int32'
		num_dims = 3
		batcher = TheanoMinibatcher(batch_size, dtype, num_dims)
		symbolic_batch = batcher.get_batch()
		updates = batcher.get_updates()

		# Mock a training function.  It just returns the current batch.
		f = function([], symbolic_batch, updates=updates)

		# Load a mock dataset
		dataset = np.array([ [[i,i,i] for j in range(2)] for i in range(num_examples)], dtype=dtype)
		actual_num_batches = batcher.load_dataset(dataset)

		# Check that we have the expected number of batches
		self.assertEqual(actual_num_batches, num_batches)

		for batch_num in range(actual_num_batches):
			batch = f()
			start, stop = batch_size*batch_num, batch_size*(batch_num+1)
			expected_batch = np.array(
				[[ [i,i,i] for j in range(2)] for i in range(start, stop)],
				dtype=dtype
			)
			self.assertTrue(np.array_equal(batch, expected_batch))



class TestWord2VecOnCorpus(TestCase):
	'''
	This tests the Word2Vec end-to-end functionality applied to a text
	corpus.
	'''

	def test_word2vec_does_pruning(self):

		files = [
			'test-data/test-corpus/003.tsv',
			'test-data/test-corpus/004.tsv'
		]

		word2vec_embedder, reader = word2vec(
			files=files,
			min_frequency=0,
			num_epochs=1,
			batch_size=int(1e2),
			macrobatch_size=int(1e5),
			t=1,
			num_embedding_dimensions=5,
			verbose=False
		)
		self.assertEqual(reader.get_vocab_size(), 303)

		word2vec_embedder, reader = word2vec(
			files=files,
			min_frequency=10,
			num_epochs=1,
			batch_size=int(1e2),
			macrobatch_size=int(1e5),
			t=1,
			num_embedding_dimensions=5,
			verbose=False
		)
		self.assertEqual(reader.get_vocab_size(), 7)


	def test_word2vec_on_corpus(self):

		# Seed randomness to make the test reproducible
		np.random.seed(1)

		word2vec_embedder, reader = word2vec(
			files=['test-data/test-corpus/numbers-long.txt'],
			num_epochs=1,
			batch_size=int(1e2),
			macrobatch_size=int(1e5),
			t=1,
			num_embedding_dimensions=5,
			verbose=False
		)

		W, C = word2vec_embedder.get_param_values()
		dots = usigma(np.dot(W,C.T))

		# Based on the construction of the corpus, the following
		# context embeddings should match the query at right and be
		# the highest value in the product of the embedding matrices
		# Note that token 0 is reserved for UNK.  It's embedding stays
		# near the randomly initialized value, tending to yield of 0.5
		# which is high overall, so it turns up as a "good match" to any
		# word
		expected_tops = [
			[2,3], # these contexts are good match to query 1
			[1,3], # these contexts are good match to query 2
			[1,2], # these contexts are good match to query 3
			[5,6], # these contexts are good match to query 4
			[4,6], # these contexts are good match to query 5
			[4,5], # these contexts are good match to query 6
			[8,9], # these contexts are good match to query 7
			[7,9], # these contexts are good match to query 8
			[7,8], # these contexts are good match to query 9
			[11,12], # these contexts are good match to query 10
			[10,12], # these contexts are good match to query 11
			[10,11]  # these contexts are good match to query 12
		]

		for i in range(1, 3*4+1):
			top3 = sorted(
				enumerate(dots[i]), key=lambda x: x[1], reverse=True
			)[:3]
			top3_positions = [t[0] for t in top3]

			# both of the expected top matches should appear in the 
			# top 3 (the UNK token might also be a good match, which
			# is why we need to check the top 3, not just top 2).
			self.assertTrue(expected_tops[i-1][0] in top3_positions)
			self.assertTrue(expected_tops[i-1][1] in top3_positions)




class TestWord2Vec(TestCase):

	'''
	This tests comnponent Word2Vec functionality by supplying
	synthetic numerical data into its components, to make sure that
	the solutions are mathematically correct.  It doesn't test iteration
	over an actual text corpus, which is tested by TestWord2VecOnCorpus.
	'''

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



	def test_save_load_embedding(self):
		'''
		Test that save and load actually write and read the 
		Word2VecEmbedder's parameters, by testing that a saved and then 
		then loaded model yields the same model as the original.
		'''

		# Remove any saved file that may be left over from a previous run
		save_dir = 'test-data/test-w2v-embedder'
		if os.path.exists(save_dir):
			shutil.rmtree(save_dir)

		# Make a Word2VecEmbedder with pre-specified query and context
		# embeddings
		mock_input = shared(np.array([[]], dtype='int32'))
		embedder = Word2VecEmbedder(
			mock_input,
			batch_size=len(self.TEST_INPUT),
			vocabulary_size=self.VOCAB_SIZE,
			num_embedding_dimensions=self.NUM_EMBEDDING_DIMENSIONS,
			query_embedding_init=self.QUERY_EMBEDDING,
			context_embedding_init=self.CONTEXT_EMBEDDING
		)
		embedder.save(save_dir)

		# Create a new embedder, and attempt to load from file
		embedder = Word2VecEmbedder(
			mock_input,
			batch_size=len(self.TEST_INPUT),
			vocabulary_size=self.VOCAB_SIZE,
			num_embedding_dimensions=self.NUM_EMBEDDING_DIMENSIONS,
		)
		embedder.load(save_dir)

		W, C = embedder.get_params()

		self.assertTrue(np.allclose(self.QUERY_EMBEDDING, W.get_value()))
		self.assertTrue(np.allclose(self.CONTEXT_EMBEDDING, C.get_value()))

		# Remove the saved file
		shutil.rmtree(save_dir)



	def test_noise_contrastive_learning(self):
		'''
		Given a simple synthetic dataset, test that the Word2vecEmbedder,
		coupled with a loss function from get_noise_contrastive_loss, 
		produces a trainable system that learns the theoretically ideal 
		embeddings as expected from [1].  

		[1] "Noise-contrastive estimation of unnormalized statistical 
			models, with applications to natural image statistics"
			by Michael U Gutmann, and Aapo Hyvarinen
		'''

		# Seed randomness to make the test reproducible
		np.random.seed(1)

		# Make the positive input.  First component of each example is
		# the query input, and second component is the context.  In the
		# final embeddings that are learned, dotting these rows and columns
		# respectively from the query and context embedding matrices should
		# give higher values than any other row-column dot products.
		signal_examples = [
			[0,2], [1,3], [2,0], [3,1], [4,6], [5,7], [6,4], [7,5], [8,9], 
			[9,8]
		]

		# Predifine the size of batches and the embedding
		num_signal_examples = len(signal_examples)
		num_noise_examples = 100
		batch_size = num_signal_examples + num_noise_examples
		vocab_size = 10
		num_embedding_dimensions = 5

		# Mock a symbolic minibatch 
		# (it's empty but will be filled in a moment)
		symbolic_batch = shared(np.array([[]], dtype='int32'))

		# Make a Word2VecEmbedder object, feed it the mocked batch
		word2vec_embedder = Word2VecEmbedder(
			input_var=symbolic_batch,
			batch_size=batch_size,
			vocabulary_size=vocab_size,
			num_embedding_dimensions=num_embedding_dimensions
		)

		# Get the params and output from the word2vec embedder
		symbolic_output = word2vec_embedder.get_output()
		params = word2vec_embedder.get_params()

		# Define the loss function, and get parameter updates based on it
		loss = get_noise_contrastive_loss(
			symbolic_output, num_signal_examples)
		updates = nesterov_momentum(
			loss, params, learning_rate=0.1, momentum=0.9)

		# Create the training function.  No inputs because the "inputs" are
		# implemented as theano shared variables
		train = function([], loss, updates=updates)

		num_replicates = 5
		num_epochs = 3000
		embedding_products = []
		W, C = word2vec_embedder.get_params()
		start = time.time()
		for rep in range(num_replicates):
			W.set_value(np.random.normal(
				0, 0.01, (vocab_size, num_embedding_dimensions)
			).astype(dtype='float32'))
			C.set_value(np.random.normal(
				0, 0.01, (vocab_size, num_embedding_dimensions)
			).astype('float32'))
			for epoch in range(num_epochs):

				# Sample new noise examples every epoch
				noise_examples = [
					[i / 10, np.random.randint(0,10)]
					for i in range(num_noise_examples)
				]

				# Assemble and load the batch to the symbolic batch variable
				batch_data = np.array(
					signal_examples + noise_examples, dtype='int32'
				)
				symbolic_batch.set_value(batch_data)

				this_loss = train()

			embedding_product = np.dot(W.get_value(), C.get_value().T)
			embedding_products.append(usigma(embedding_product))

		mean_embedding_products = np.mean(embedding_products, axis=0)

		# We expect that the embeddings will allocate the most probability
		# to the contexts that were provided for words in the toy data.
		# We always provided a single context via batch_contexts
		# (e.g. context 2 provided for word 0), so we expect these contexts
		# to be the maximum.
		expected_max_prob_contexts = np.array(signal_examples)[:,1]
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


	def test_Word2VecEmbedder(self):
		'''
		Test that the architecture defined by Word2VecEmbedder generates the
		expected theano expressions for embedded inputs and for final output
		'''

		input_var = T.imatrix('input_var')

		embedder = Word2VecEmbedder(
			input_var,
			batch_size=len(self.TEST_INPUT),
			vocabulary_size=self.VOCAB_SIZE,
			num_embedding_dimensions=self.NUM_EMBEDDING_DIMENSIONS,
			query_embedding_init=self.QUERY_EMBEDDING,
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
