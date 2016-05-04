from unigram_dictionary import UnigramDictionary
from collections import Counter, defaultdict
from token_map import TokenMap, SILENT, ERROR
import time
from unittest import main, TestCase
from theano import tensor as T, function
import numpy as np
from w2v import Word2VecEmbedder, word2vec
from noise_contrast import noise_contrast, NoiseContraster
from minibatcher import Word2VecMinibatcher, TokenChooser, default_parse
from counter_sampler import MultinomialSampler, CounterSampler
from lasagne.init import Normal


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


	# TODO test that we actually produce the samples according to the 
	# frequency distribution of the corpus
	def test_sampling(self):
		'''
		Test basic function of assigning counts, and then sampling from
		The distribution implied by those counts.
		'''
		unigram_dictionary = UnigramDictionary()
		unigram_dictionary.update(self.FREQUENCIES)

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


class TestMultinomialSampler(TestCase):

	def test_multinomial_sampler(self):
		counts = range(1,6)
		sampler = MultinomialSampler(counts)

		# Test asking for a single sample (where no shape tuple supplied)
		single_sample = sampler.sample()
		self.assertTrue(type(single_sample) is np.int64)

		# Test asking for an array of samples (by passing a shape tuple)
		shape = (2,3,5)
		array_sample = sampler.sample(shape)
		self.assertTrue(type(array_sample) is np.ndarray)
		self.assertTrue(array_sample.shape == shape)


	def test_multinomial_sampler_stats(self):
		'''
		This tests that the sampler really does produce results whose
		statistics match those requested by the counts vector
		'''
		# Seed numpy's random function to make the test reproducible
		np.random.seed(1)

		# Make a sampler with probabilities proportional to counts
		counts = range(1,6)
		sampler = MultinomialSampler(counts)

		# Draw one hundred thousand samples, then total up the fraction of
		# each outcome obseved
		counter = Counter(sampler.sample((100000,)))
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

		#print 'expected_loss:'
		#print expected_loss
		#print
		#print 'test_loss:'
		#print test_loss
		#print

		self.assertAlmostEqual(test_loss, expected_loss)



class TestMinibatcher(TestCase):

	def setUp(self):

		# Define some parameters to be used in construction 
		# Minibatcher
		self.files = [
			'test-data/test-corpus/003.tsv',
			'test-data/test-corpus/004.tsv'
		]
		self.batch_size = 5
		self.noise_ratio = 15
		self.t = 0.03

		# Make a minibatch generator
		self.generator = Word2VecMinibatcher(
			files=self.files,
			t=self.t,
			batch_size=self.batch_size,
			noise_ratio=self.noise_ratio,
			verbose=False
		)

		# Make another Word2VecMinibatcher, and pre-load this one with 
		# token_map and counter_sampler distribution information.
		self.preloaded_generator = Word2VecMinibatcher(
			files=self.files,
			t=self.t,
			batch_size=self.batch_size,
			noise_ratio=self.noise_ratio,
			verbose=False
		)
		self.preloaded_generator.load('test-data/minibatch-generator-test')


	def test_prepare(self):
		'''
		Check that Word2VecMinibatcher.prepare() properly makes a 
		UnigramDictionary that reflects the corpus.
		'''
		self.generator.prepare()
		d = self.generator.unigram_dictionary

		# Make sure that all of the expected tokens are found in the 
		# unigram_dictionary, and that their frequency in the is correct.
		tokens = []
		for filename in self.files:
			for add_tokens in default_parse(filename):
				tokens.extend(add_tokens)

		counts = Counter(tokens)
		for token in tokens:
			token_id = d.get_id(token)
			count = d.get_frequency(token_id)
			self.assertEqual(count, counts[token])


	def test_minibatches(self):
		'''
		Make sure that the minibatches are the correct size, that 
		signal query- and contexts-words are always within 5 tokens of
		one another and come from the same sentence.
		'''
		# Ensure reproducibility in this stochastic test
		np.random.seed(1)

		# Before looking at the minibatches, we need to determine what 
		# query-context pairs are possible.
		# To do that, first read in the corpus, and break it into lines
		# and tokens
		lines = []

		# Go through the corpus and get all the token ids as a list
		tokenized_lines = []
		for filename in self.files:
			for tokens in default_parse(filename):
				tokenized_lines.append(tokens)

		# Now iterate through the lines, noting what tokens arise within
		# one another's contexts.  Build a lookup table providing the set
		# of token_ids that arose in the context of each given token_id
		legal_pairs = defaultdict(set)
		d = self.preloaded_generator.unigram_dictionary
		for line in tokenized_lines:
			token_ids = d.get_ids(line)
			for i, token_id in enumerate(token_ids):
				low = max(0, i-5)
				legal_context = token_ids[low:i] + token_ids[i+1:i+6]
				legal_pairs[token_id].update(legal_context)

		# finally, add UNK to the legal pairs
		legal_pairs[0] = set([0])

		for signal_batch, noise_batch in self.preloaded_generator:

			self.assertEqual(len(signal_batch), self.batch_size)
			self.assertEqual(
				len(noise_batch), self.batch_size * self.noise_ratio
			)

			# Ensure that all of the signal examples are actually valid
			# samples from the corpus
			for query_token_id, context_token_id in signal_batch:
				self.assertTrue(
					context_token_id in legal_pairs[query_token_id]
				)

	def test_token_discarding(self):

		# Ensure reproducibility for the test
		np.random.seed(1)

		# Get the preloaded generator and its unigram_dictionary
		self.preloaded_generator
		d = self.preloaded_generator.unigram_dictionary

		# Go through the corpus and get all the token ids as a list
		token_ids = []
		for filename in self.files:
			for tokens in default_parse(filename):
				token_ids.extend(d.get_ids(tokens))

		# Run through the tokens, evaluating 
		# Word2VecMinibatcher.do_discard() on each.  Keep track of all 
		# "discarded" tokens for which do_discard() returns True
		discarded = []
		num_replicates = 100
		for replicates in range(num_replicates):
			for token_id in token_ids:
				if self.preloaded_generator.do_discard(token_id):
					discarded.append(token_id)

		# Count the tokens, and the discarded tokens.  
		discarded_counts = Counter(discarded)
		token_counts = Counter(token_ids)

		# Compute the frequency of the word "the", and the frequency
		# with which it was discarded
		the_id = d.get_id('the')
		num_the_appears = token_counts[the_id]
		the_frequency = num_the_appears/float(len(token_ids))
		num_the_discarded = discarded_counts[the_id]
		frequency_of_discard = (
			num_the_discarded / float(num_the_appears * num_replicates)
		)

		# What was actually the most discarded token?  It should be "the"
		most_discarded_id, num_most_discarded = (
			discarded_counts.most_common()[0]
		)
		self.assertEqual(most_discarded_id, the_id)

		# What was the expected frequency with which "the" would be 
		# discarded?  Assert it is close to the actual discard rate.
		expected_frequency = 1 - np.sqrt(self.t / the_frequency)
		tolerance = 0.015
		self.assertTrue(
			abs(expected_frequency - frequency_of_discard) < tolerance
		)



class TestWord2VecOnCorpus(TestCase):
	'''
	This tests the Word2Vec end-to-end functionality applied to a text 
	corpus.
	'''

	def test_word2vec_on_corpus(self):

		# Seed randomness to make the test reproducible
		np.random.seed(1)

		word2vec_embedder, dictionary = word2vec(
			files=['test-data/test-corpus/numbers-long.txt'],
			num_epochs=1,
			t=1, 
			batch_size = 10,
			num_embedding_dimensions=5,
			verbose=False,
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
			[0,2,3], # these contexts are good match to query 1
			[0,1,3], # these contexts are good match to query 2 
			[0,1,2], # these contexts are good match to query 3 
			[0,5,6], # these contexts are good match to query 4 
			[0,4,6], # these contexts are good match to query 5 
			[0,4,5], # these contexts are good match to query 6 
			[0,8,9], # these contexts are good match to query 7 
			[0,7,9], # these contexts are good match to query 8 
			[0,7,8], # these contexts are good match to query 9 
			[0,11,12], # these contexts are good match to query 10
			[0,10,12], # these contexts are good match to query 11 
			[0,10,11]  # these contexts are good match to query 12 
		]

		for i in range(1, 3*4+1):
			top3 = sorted(
				enumerate(dots[i]), key=lambda x: x[1], reverse=True
			)[:3]
			top3_positions = [t[0] for t in top3]
			self.assertItemsEqual(top3_positions, expected_tops[i-1])

		print repr(word2vec_embedder.embed([1,2,3,4]))




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


	# TODO: finish this
	def test_save_load(self):

		input_var = T.imatrix('input_var')

		embedder = Word2VecEmbedder(
			input_var,
			batch_size=len(self.TEST_INPUT),
			vocabulary_size=self.VOCAB_SIZE,
			num_embedding_dimensions=self.NUM_EMBEDDING_DIMENSIONS,
			word_embedding_init=self.QUERY_EMBEDDING,
			context_embedding_init=self.CONTEXT_EMBEDDING
		)

		W, C = embedder.get_param_values()


	def test_noise_contrastive_learning(self):

		# Seed randomness to make the test reproducible
		np.random.seed(1)

		# Predifine the size of batches and the embedding
		batch_size = 160
		vocab_size = 10
		num_embedding_dimensions = 5

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
			num_embedding_dimensions,
		)

		# Get the params and output from the word2vec embedder, feed that
		# back to the noise_contraster to get the training function
		combined_output = word2vec_embedder.get_output()
		params = word2vec_embedder.get_params()
		train = noise_contraster.get_train_func(combined_output, params)


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
		W, C = word2vec_embedder.get_params()
		start = time.time()
		for rep in range(num_replicates):
			W.set_value(np.random.normal(
				0, 0.01, (vocab_size, num_embedding_dimensions)
			).astype(dtype='float32'))
			C.set_value(np.random.normal(
				0, 0.01, (vocab_size, num_embedding_dimensions)
			).astype('float32'))
			#print '\t***'
			for epoch in range(num_epochs):

				# Sample new noise examples every epoch (this is better 
				# than fixing the noise once at the start).
				# Provide 15 negative examples for each query word
				test_negative_input = np.array([
					[i / 10, np.random.randint(0,10)] for i in range(100)
				]).astype('int32')

				loss = train(
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
		#print 'average weight to correct pairs:', np.mean(
		#	embedding_maxima
		#)
		#print 'elapsed time:', time.time() - start



	# TODO: re-implement this test without the defunct Word2Vec class
	#def test_learning(self):

	#	positive_input = T.imatrix('query_input')
	#	negative_input = T.imatrix('noise_input')

	#	# Predifine the size of batches and the embedding
	#	batch_size = 160
	#	vocab_size = 10
	#	num_embedding_dimensions = 5

	#	# Make a Word2Vec object
	#	word2vec = Word2Vec(
	#		batch_size,
	#		vocabulary_size=vocab_size,
	#		num_embedding_dimensions=num_embedding_dimensions,
	#		learning_rate=0.1,
	#		verbose=False
	#	)

	#	# Make the positive input.  First component of each example is
	#	# the query input, and second component is the context.  In the 
	#	# final embeddings that are learned, dotting these rows and columns
	#	# respectively from the query and context embedding matrices should
	#	# give higher values than any other row-column dot products.
	#	test_positive_input = np.array([
	#		[0,2],
	#		[1,3],
	#		[2,0],
	#		[3,1],
	#		[4,6],
	#		[5,7],
	#		[6,4],
	#		[7,5],
	#		[8,9],
	#		[9,8]
	#	]).astype('int32')
	#	
	#	num_replicates = 5
	#	num_epochs = 3000
	#	embedding_products = []
	#	W, C = word2vec.get_params()
	#	start = time.time()
	#	for rep in range(num_replicates):
	#		W.set_value(np.random.normal(
	#			0, 0.01, (vocab_size, num_embedding_dimensions)
	#		).astype(dtype='float32'))
	#		C.set_value(np.random.normal(
	#			0, 0.01, (vocab_size, num_embedding_dimensions)
	#		).astype('float32'))
	#		#print '\t***'
	#		for epoch in range(num_epochs):

	#			# Sample new noise examples every epoch (this is better than
	#			# fixing the noise once at the start).
	#			# Provide 15 negative examples for each query word
	#			test_negative_input = np.array([
	#				[i / 10, random.randint(0,9)] for i in range(100)
	#			]).astype('int32')

	#			loss = word2vec.train(
	#				test_positive_input, test_negative_input
	#			)
	#			#print loss

	#		embedding_product = np.dot(W.get_value(), C.get_value().T)
	#		embedding_products.append(usigma(embedding_product))

	#	mean_embedding_products = np.mean(embedding_products, axis=0)
	#	#print np.round(mean_embedding_products, 2)

	#	# We expect that the embeddings will allocate the most probability
	#	# to the contexts that were provided for words in the toy data.
	#	# We always provided a single context via batch_contexts 
	#	# (e.g. context 2 provided for word 0), so we expect these contexts
	#	# to be the maximum.
	#	expected_max_prob_contexts = test_positive_input[:,1]
	#	self.assertTrue(np.array_equal(
	#		np.argmax(mean_embedding_products, axis=1),
	#		expected_max_prob_contexts
	#	))

	#	# The dot product of a given word embedding and context embedding
	#	# have an interpretation as the probability that that word and
	#	# context derived from the toy data instead of the noise.
	#	# See equation 3 in Noise-Contrastive Estimation of Unnormalized 
	#	# Statistical Models, with Applications to Natural Image 
	#	# StatisticsJournal of Machine Learning Research 13 (2012), 
	#	# pp.307-361.
	#	# That shows the probability should be around 0.5
	#	# Since the actual values are stocastic, we check that the 
	#	# average of repeated trials is within 0.25 - 0.75.
	#	embedding_maxima = np.max(mean_embedding_products, axis=1)
	#	self.assertTrue(all(
	#		[x > 0.25 for x in embedding_maxima]
	#	))
	#	self.assertTrue(all(
	#		[x < 0.75 for x in embedding_maxima]
	#	))
	#	#print 'average weight to correct pairs:', np.mean(
	#	#	embedding_maxima
	#	#)
	#	#print 'elapsed time:', time.time() - start


	# TODO: re-implement this test without the defunct Word2Vec class
	#def test_positive_negative_separation(self):

	#	vocabulary_size=100000,
	#	num_embedding_dimensions=500,
	#	word_embedding_init=Normal(), 
	#	context_embedding_init=Normal(),

	#	embedder = Word2Vec(
	#		batch_size=len(self.TEST_INPUT),
	#		vocabulary_size=self.VOCAB_SIZE,
	#		num_embedding_dimensions=self.NUM_EMBEDDING_DIMENSIONS,
	#		word_embedding_init=self.QUERY_EMBEDDING,
	#		context_embedding_init=self.CONTEXT_EMBEDDING
	#	)

	#	# Get Word2Vec's internal symbolic theano variables
	#	positive_input = embedder.positive_input
	#	negative_input = embedder.negative_input
	#	positive_output = embedder.positive_output
	#	negative_output = embedder.negative_output

	#	# Compile the function connecting the symbolic inputs and outputs
	#	# So that we can test the separation of the signal and noise 
	#	# channels after they are processed by the underlying 
	#	# Word2VecEmbedder
	#	f = function(
	#		[positive_input, negative_input],
	#		[positive_output, negative_output]
	#	)

	#	# Calculate the positive and negative output
	#	test_positive_input = self.TEST_INPUT[:6]
	#	test_negative_input = self.TEST_INPUT[6:]
	#	test_positive_output, test_negative_output = f(
	#		test_positive_input, test_negative_input
	#	)

	#	# Calculate the expected embeddings and output
	#	expected_query_embeddings = np.repeat(
	#		self.QUERY_EMBEDDING, 3, axis=0
	#	)
	#	expected_context_embeddings = np.tile(
	#		self.CONTEXT_EMBEDDING, (3,1)
	#	)
	#	expected_output = usigma(np.dot(
	#		expected_query_embeddings, expected_context_embeddings.T
	#	)).diagonal()
	#	expected_positive_output = expected_output[:6]
	#	expected_negative_output = expected_output[6:]

	#	#print 'test_positive_output:'
	#	#print test_positive_output
	#	#print
	#	#print 'expected_positive_output:'
	#	#print expected_positive_output
	#	#print
	#	#print 'test_negative_output:'
	#	#print test_negative_output
	#	#print 
	#	#print 'expected_negative_output:'
	#	#print expected_negative_output
	#	#print 

	#	# Check for equality between all found and expected values
	#	self.assertTrue(np.allclose(
	#		test_positive_output, expected_positive_output
	#	))
	#	self.assertTrue(np.allclose(
	#		test_negative_output, expected_negative_output
	#	))


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
	
