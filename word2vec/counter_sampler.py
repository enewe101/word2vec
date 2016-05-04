'''
Calculates the maximum likelihood probability of single words, as equal
to their frequency within a corpus, and enables efficiently sampling
from the distribution.

Assumes that the words have been converted into ids that were assigned
by auto-incrementing from 0 

(The consequence if that is not the case isn't huge, it just means memory
would be wasted)
'''

import numpy as np
import gzip


class CounterSamplerException(Exception):
	pass


class CounterSampler(object):

	def __init__(self, counts=[]):
		'''
		Create a CounterSampler.  Most common usage is to provide no
		arguments, creating an empty CounterSampler, because CounterSampler 
		provides functions to accumulate counts from observations.

		counts:	list of counts.  The index of the count in the counts list
			serves as the id for the outcome having that many counts.
		'''

		# Validation
		if not all([isinstance(c, int) for c in counts]):
			raise ValueError('Counts must be an iterable of integers')
		self.counts = list(counts)
		self.sampler_ready = False


	def add(self, idx):
		self.sampler_ready = False
		if len(self.counts) <= idx:
			add_zeros = [0] * (idx +1 - len(self.counts))
			self.counts.extend(add_zeros)

		self.counts[idx] += 1


	def update(self, idxs):
		for idx in idxs:
			self.add(idx)


	def get_sampler(self):
		if len(self.counts) < 1:
			raise CounterSamplerException(
				'Cannot sample if no counts have been made'
			)

		if not self.sampler_ready:
			self._sampler = MultinomialSampler(self.counts)
			self.sampler_ready = True

		return self._sampler


	def get_total_counts(self):
		return self.get_sampler().total
	def __len__(self):
		return self.get_sampler().total


	def sample(self, shape=()):
		'''
		Draw a sample according to the counter_sampler probability
		'''

		return self.get_sampler().sample(shape)


	def save(self, filename):
		if filename.endswith('.gz'):
			f = gzip.open(filename, 'w')
		else:
			f = open(filename, 'w')	
		
		for c in self.counts:
			f.write('%d\n' % c)


	def load(self, filename):
		if filename.endswith('.gz'):
			f = gzip.open(filename)
		else:
			f = open(filename)

		self.counts = [int(c) for c in f.readlines()]


	def get_frequency(self, idx):
		'''
		Return the number of observations for outcome idx.
		'''
		return self.counts[idx]

	def get_probability(self, token_id):
		'''
		Return the probability associated to token_id.
		'''
		# Delegate to the underlying MultinomialSampler
		return self.get_sampler().get_probability(token_id)



class MultinomialSampler(object):

	def __init__(self, scores):
		if len(scores) < 1:
			raise ValueError('The scores list must have length >= 1')
		self.scores = scores
		self.total = float(sum(scores))
		self.K = len(scores)
		self.setup()


	def get_probability(self, k):
		'''
		Get the actual probability associated to outcome k
		'''
		return self.orig_prob_mass[k]


	def setup(self):
		self.orig_prob_mass = np.zeros(self.K)
		self.mixture_prob_mass = np.zeros(self.K)
		self.mass_reasignments = np.zeros(self.K, dtype=np.int64)
 
		# Sort the data into the outcomes with probabilities
		# that are larger and smaller than 1/K.
		smaller = []
		larger  = []
		for k, score in enumerate(self.scores):
			self.orig_prob_mass[k] = score / self.total
			self.mixture_prob_mass[k] = (self.K*score) / self.total
			if self.mixture_prob_mass[k] < 1.0:
				smaller.append(k)
			else:
				larger.append(k)
 
		# We will have k different slots. Each slot represents 1/K
		# prbability mass, and to each we allocate all of the probability
		# mass from a "small" outcome, plus some probability mass from
		# a "large" outcome (enough to equal the total 1/K).
		# We keep track of the remaining mass left for the larger outcome,
		# allocating the remainder to another slot later.
		# The result is that the kth has some mass allocated to outcome
		# k, and some allocated to another outcome, recorded in J[k].
		# q[k] keeps track of how much mass belongs to outcome k, and 
		# how much belongs to outcome J[k].
		while len(smaller) > 0 and len(larger) > 0:
			small_idx = smaller.pop()
			large_idx = larger.pop()
 
			self.mass_reasignments[small_idx] = large_idx
			self.mixture_prob_mass[large_idx] = (
				self.mixture_prob_mass[large_idx] -
				(1.0 - self.mixture_prob_mass[small_idx])
			)
 
			if self.mixture_prob_mass[large_idx] < 1.0:
				smaller.append(large_idx)
			else:
				larger.append(large_idx)
 
		return self.mass_reasignments, self.mixture_prob_mass


	def sample(self, shape=()):

		if len(shape) < 1:
			return self._sample()
		else:
			this_dim = shape[0]
			recurse_shape = shape[1:]
			return np.array(
				[self.sample(recurse_shape) for i in range(this_dim)]
			, dtype='int64')


	def _sample(self):

		# Draw from the overall uniform mixture.
		k = np.int64(int(np.floor(np.random.rand()*self.K)))
	 
		# Draw from the binary mixture, either keeping the
		# small one, or choosing the associated larger one.
		if np.random.rand() < self.mixture_prob_mass[k]:
			return k
		else:
			return self.mass_reasignments[k]

