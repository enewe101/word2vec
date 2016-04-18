from multiprocessing import Queue, Process, Pipe
from Queue import Empty
from corpus_reader import CorpusReader
from unigram import Unigram, MultinomialSampler
from dictionary import Dictionary
import numpy as np
import gzip
import os


class MinibatchGenerator(object):

	NOT_DONE = 0
	DONE = 1

	def __init__(
		self,
		files=[],
		directories=[],
		skip=[],
		dictionary=None,
		unigram=None,
		noise_ratio=15,
		kernel=[1,2,3,4,5,5,4,3,2,1],
		t = 1.0e-5,
		batch_size = 1000
	):

		# Get a corpus reader
		self.corpus_reader = CorpusReader(files, directories, skip)

		# Load the dictionary, if supplied
		if dictionary is not None:
			self.dictionary = dictionary
		else:
			self.dictionary = Dictionary()

		# Load the unigram distribution, if supplied
		if unigram is not None:
			self.unigram = unigram
		else:
			self.unigram = Unigram()

		self.noise_ratio = noise_ratio
		self.kernel = kernel
		self.t = t
		self.batch_size = batch_size

		# Validate the kernel.  It should reflect the relative 
		# frequencies of choosing tokens from a window of +/- K tokens
		# relative to a query token.  So it must have an even number of
		# entries
		if not len(self.kernel) % 2 == 0:
			raise ValueError(
				'kernel should reflect the relative frequencies of '
				'selecting a context token within +/- K of the query '
				'token, and so should have an equal number of entries '
				'defining frequencies to the left and right of the query '
				'token, and so should have an even number of entries.'
			)


	def get_vocab_size(self):
		'''
		Get the size of the vocabulary.  Only makes sense to call this
		after MinibatchGenerator.prepare() has been called, or if an
		existing (pre-filled) dictionary was loaded, since otherwise 
		it would just return 0.  Delegate to the underlying dictionary.
		'''
		return self.dictionary.get_vocab_size()


	def load(self, directory):
		'''
		Load both the dictionary and unigram, assuming default filenames
		(dictionary.gz and unigram.gz), by specifying their containing
		directory
		'''
		self.dictionary.load(os.path.join(directory, 'dictionary.gz'))
		self.unigram.load(os.path.join(directory, 'unigram.gz'))

	
	def load_dictionary(self, filename):
		self.dictionary.load(filename)


	def load_unigram(self, filename):
		self.unigram.load(filename)


	def save(self, directory):
		'''
		Save both the dictionary and unigram, using default filenames
		(dictionary.gz and unigram.gz), by specifying only their containing
		directory
		'''
		self.dictionary.save(os.path.join(directory, 'dictionary.gz'))
		self.unigram.save(os.path.join(directory, 'unigram.gz'))


	def save_dictionary(self, filename):
		self.dictionary.save(filename)


	def save_unigram(self, filename):
		self.unigram.save(filename)


	def prepare(self):
		# Before any minibatches can be generated, we need to run over
		# the corpus to determine the unigram distribution and create
		# a dictionary mapping all words in the corpus vocabulary to int's.
		for line in self.corpus_reader.read():
			token_ids = self.dictionary.update(line)
			self.unigram.update(token_ids)


	def __iter__(self):

		# Once iter is called, a subprocess will be started which
		# begins generating minibatches.  These accumulate in a queue
		# and iteration pulls from that queue.  That way, iteration
		# can begin as soon as the first minibatch is prepared, and 
		# later minibatches are prepared in the background while earlier
		# minibatches are used.  The idea is that this will keep the 
		# CPU(s) busy while training occurs on the GPU

		# Read through the corpus line by line, and generate samples
		self.minibatches = Queue()
		self.recv_pipe, send_pipe = Pipe()

		# We'll fork a process to assemble minibatches, and return 
		# immediatetely so that minibatches can be used as they are 
		# constructed.
		#
		# Because we assemble the batches within a forked process, it's 
		# access to randomness doesn't alter the state of the parent's 
		# random number generator.  Multiple calls to this function
		# would produce the same set of random samples, which is not
		# desired.  We make a call to the numpy random number generator
		# to advance the parent's random number generator's state to avoid
		# this problem:
		np.random.uniform()

		minibatch_preparation = Process(
			target=self.prepare_minibatches,
			args=(self.minibatches, send_pipe)
		)
		minibatch_preparation.start()

		return self


	def init_batch(self):
		# Initialize np.array's to store the minibatch data.  We know
		# how big the batch is ahead of time.  Initialize by filling
		# the arrays with UNK tokens.  Doing this means that, at the end
		# of the corpus, when we don't necessarily have a full minibatch,
		# the final minibatch is padded with UNK tokens in order to be
		# of the desired shape.  This has no effect on training, because
		# we don't care about the embedding of the UNK token
		signal_batch = np.full(
			(self.batch_size, 2),
			self.dictionary.UNK,
			dtype='int32'
		)
		noise_batch = np.full(
			(self.batch_size * self.noise_ratio, 2),
			self.dictionary.UNK,
			dtype='int32'
		)
		return signal_batch, noise_batch


	def prepare_minibatches(self, minibatch_queue, send_pipe):

		chooser = TokenChooser(K=len(self.kernel)/2, kernel=self.kernel)
		signal_batch, noise_batch = self.init_batch()

		# i keeps track of position in the signal batch
		i = -1
		for line in self.corpus_reader.read():

			# Isolated tokens (e.g. one-word sentences) have no context
			# and can't be used for training.
			if len(line) < 2:
				continue

			token_ids = self.dictionary.get_ids(line)

			# We'll now generate generate signal examples and noise
			# examples for training
			for query_token_pos, query_token_id in enumerate(token_ids):

				# Possibly discard the token
				if self.do_discard(query_token_id):
					continue

				# Increment position within the batch
				i += 1

				# Sample a token from the context
				context_token_pos = chooser.choose_token(
					query_token_pos, len(token_ids)
				)
				context_token_id = token_ids[context_token_pos]
				signal_batch[i, :] = [query_token_id, context_token_id]

				# Sample tokens from the noise
				noise_context_ids = self.unigram.sample((self.noise_ratio,))

				# Figure out the position within the noise batch
				j = i*self.noise_ratio

				# block-assign the noise samples to the noise batch array
				noise_batch[j:j+self.noise_ratio, :] = [
					[query_token_id, noise_context_id]
					for noise_context_id in noise_context_ids
				]

				# Once we've finished assembling a minibatch, enqueue it
				# and start assemblin a new minibatch
				if i == self.batch_size - 1:
					minibatch_queue.put((signal_batch, noise_batch))
					signal_batch, noise_batch = self.init_batch()
					i = -1

		# Normally we'll have a partially filled minibatch after processing
		# the corpus.  The elements in the batch that weren't overwritten
		# contain UNK tokens, which act as padding.  Enqueue the partial
		# minibatch.
		if i >= 0:
			minibatch_queue.put((signal_batch, noise_batch))

		send_pipe.send(self.DONE)


	def do_discard(self, token_id):
		'''
		This function helps with downsampling of very common words.
		Returns true when the token should be discarded as a query word
		'''
		probability = self.unigram.get_probability(token_id)
		discard_probability = 1 - np.sqrt(self.t/probability)
		do_discard = np.random.uniform() < discard_probability

		#if do_discard:
		#	print 'discarding', self.dictionary.get_token(token_id)

		return do_discard


	def next(self):
		status = self.NOT_DONE
		while status == self.NOT_DONE:
			try:
				return self.minibatches.get(timeout=0.1)
			except Empty:
				if self.recv_pipe.poll():
					status = self.recv_pipe.recv()

		raise StopIteration

				

class TokenChooser(object):

	'''
	This choses which context token should be taken given a window
	of +/- K around a query token
	'''

	def __init__(self, K, kernel):
		if not len(kernel) == 2*K:
			raise ValueError(
				'`kernel` must have 2*K entries, one for '
				'each of the elements within the windows of +/- K tokens.'
			)

		self.K = K
		self.kernel = kernel
		self.samplers = {}
		self.indices = range(-K, 0) + range(1, K+1)


	def choose_token(self, idx, length):
		'''
		Randomly choosea  token according to the kernel supplied
		in the constructor.  Note that when sampling the context near
		the beginning of a sentence, the left part of the context window
		will be truncated.  Similarly, sampling context near the end of
		a sentence leads to truncation of the right part of the context 
		window.  Short sentences lead to truncation on both sides.

		To ensure that samples are returned within the possibly truncated
		window, two values define the actual extent of the context to be
		sampled:

		`idx`: index of the query word within the context.  E.g. if the
			valid context is constrained to a sentence, and the query word
			is the 3rd token in the sentence, idx should be 2 (because
			of 0-based indexing)

		`length`: length of the the context, E.g. If context is 
			constrained to a sentence, and sentence is 7 tokens long,
			length should be 7.
		'''

		# If the token is near the edges of the context, then the 
		# sampling kernel will be truncated (we can't sample before the 
		# firs word in the sentence, or after the last word).
		# Determine the slice indices that define the truncated kernel.
		negative_idx = length - idx
		start = max(0, self.K - idx)
		stop = min(2*self.K, self.K + negative_idx - 1)	

		# We make a separate multinomial sampler for each different 
		# truncation of the kernel, because they each define a different 
		# set of sampling probabilityes.  If we don't have a sampler for 
		# this particular kernel shape, make one.
		if not (start, stop) in self.samplers:

			trunc_probabilities = self.kernel[start:stop]
			self.samplers[start,stop] = (
				MultinomialSampler(trunc_probabilities)
			)

		# Sample from the multinomial sampler for the context of this shape
		outcome_idx = self.samplers[start,stop].sample()

		# Map this into the +/- indexing relative to the query word
		relative_idx = self.indices[outcome_idx + start]

		# And then map this into absolute indexing
		result_idx = relative_idx + idx

		return result_idx


