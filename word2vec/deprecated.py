import os 
from multiprocessing import Process, Pipe, Queue
from Queue import Empty
import sys


def default_parse(filename):
	tokenized_sentences = []
	for line in open(filename):
		tokenized_sentences.append(line.strip().split())
	return tokenized_sentences



class CorpusReader(object):

	NOT_DONE = 0
	DONE = 1

	def __init__(
		self, 
		files=[],
		directories=[],
		skip=[],
		parse=default_parse,
		verbose=True
	):
		self.files = files
		self.directories = directories
		self.skip = skip
		self.parse = parse
		self.verbose = verbose


	def read_no_q(
		self
	):
		'''
		Iterates through the files and directories given in the constructor
		and parses out a list of sentences, where sentences are encoded
		as a list of tokens (i.e. a list of lists of tokens).
		Parsing the files is deligated to a parser function, which can
		be customized.

		Lines are loaded into a queue so that reading can be done in the
		background.
		'''

		# Process all the files listed in files, unles they match an
		# entry in skip
		#print 'starting reading'
		if self.files is not None:
			for filename in self.files:
				filename = os.path.abspath(filename)

				# Skip files if they match a regex in skip
				if any([s.search(filename) for s in self.skip]):
					continue

				if self.verbose:
					print 'processing', filename
				for line in self.parse(filename):
					yield(line)

		# Process all the files listed in each directory, unless they
		# match an entry in skip
		if self.directories is not None:
			for dirname in self.directories:
				dirname = os.path.abspath(dirname)

				# Skip directories if they match a regex in skip
				if any([s.search(dirname) for s in self.skip]):
					continue

				for filename in os.listdir(dirname):
					filename = os.path.join(dirname, filename)

					# Only process the *files* under the given directories
					if not os.path.isfile(filename):
						continue

					# Skip files if they match a regex in skip
					if any([s.search(filename) for s in self.skip]):
						continue

					if self.verbose:
						print 'processing', filename
					for line in self.parse(filename):
						yield line



	def read(self):

		queue = Queue()
		pipe1, pipe2 = Pipe()

		# We do the reading in a separate process, that way, if the
		# consumer is busy processing the read items, we keep loading
		# the corpus in the background
		reading_process = Process(
			target=self._read, 
			args=(self.files, self.directories, self.skip, queue, pipe2)
		)
		reading_process.start()

		state = self.NOT_DONE

		while state == self.NOT_DONE:

			try:
				yield queue.get(timeout=0.1)

			except Empty:
				if pipe1.poll():
					state = pipe1.recv()


	def _read(
		self,
		files=[],
		directories=[],
		skip=[],
		queue=None,
		pipe=None
	):
		'''
		Iterates through the files and directories given in the constructor
		and parses out a list of sentences, where sentences are encoded
		as a list of tokens (i.e. a list of lists of tokens).
		Parsing the files is deligated to a parser function, which can
		be customized.

		Lines are loaded into a queue so that reading can be done in the
		background.
		'''

		# Process all the files listed in files, unles they match an
		# entry in skip
		if files is not None:
			for filename in files:
				filename = os.path.abspath(filename)

				# Skip files if they match a regex in skip
				if any([s.search(filename) for s in skip]):
					continue

				for line in self.parse(filename):
					queue.put(line)

		# Process all the files listed in each directory, unless they
		# match an entry in skip
		if directories is not None:
			for dirname in directories:
				dirname = os.path.abspath(dirname)

				# Skip directories if they match a regex in skip
				if any([s.search(dirname) for s in skip]):
					continue

				for filename in os.listdir(dirname):
					filename = os.path.join(dirname, filename)

					# Only process the *files* under the given directories
					if not os.path.isfile(filename):
						continue

					# Skip files if they match a regex in skip
					if any([s.search(filename) for s in skip]):
						continue

					for line in self.parse(filename):
						queue.put(line)

		# Notify the parent process that you're done
		if pipe is not None:
			pipe.send(self.DONE)






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
		1) Makes a token_map which encodes tokens as integers
			- the token_map class stores a two-way mapping between 
				token types (i.e. unique words) and integers
			- forward mapping from tokens to integers uses a token_map
				with keys as tokens and ints as values
			- revrese mapping from integers to tokens uses a list of tokens
		2) Makes a counter_sampler model which enables sampling from the counter_sampler
			model
			- keeps count of how many times a given tokens were seen
				(using a list whose values are counts and whose 
					components are the token ids)
			- enables rapid sampling from the counter_sampler distribution. Using
				a binary tree structure.  The first time sampling is 
				attempted after an update has been made to the counts,
				the tree is updated / generated.
			- a full pass through the corpus is needed to create the counter_sampler
				model.
		3) Generates batches of signal and noise examples to train the
			embeddings


		If `savedir` is specified, then three files will be saved into
		the directory specified by savedir (savedir will be created if 
		it doesn't exist, as long as other dirs in its path already exist)
		1) savedir/token_map.gz -- stores the token_map mapping
		2) savedir/counter_sampler.gz -- stores the counter_sampler frequencies.  This 
			means that future training using different sampling will
			not need to re-count the counter_sampler frequencies!
		3) savedir/embeddings.gz -- stores the embedding parameters for
			both query-embeddings (which is the main word embedding of use
			in other applications) as well as the context-embedding (not
			usually needed for other applications, but kept for 
			completeness)

		if `loaddir` is specified, the token_map and counter_sampler saved in
		loaddir/token_map.gz and loaddir/counter_sampler.gz will be loaded.
		This means that the counter_sampler frequencies (and token_map) don't 
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

		# Create a token_map.  We'll pass it into the minibatch generator,
		# but we want a reference to it in the Word2Vec object too, 
		# since it also needs the token_map
		token_map = TokenMap()

		# Make a minibatch generator
		minibatch_generator = Word2VecMinibatcher(
			files, directories, skip, token_map
		)

		# Run the minibatch generator over the corpus to collect counter_sampler
		# statistics and to fill out the token_map
		minibatch_generator.prepare_counter_sampler()

		# We now have a full token_map and the counter_sampler statistics.
		# Save them if savedir was specified
		if savedir is not None:
			token_map.save(os.path.join(savedir, 'token_map.gz'))
			minibatch_generator.save_counter_sampler(
				os.path.join(savedir, 'counter_sampler.gz')
			)

		# Here is where the training actually happens
		for positive_input, negative_input in minibatch_generator:
			self.train(positive_input, negative_input)

		# Save the learned embedding (if savedir was specified)
		if savedir is not None:
			self.embedder.save(os.path.join(savedir, 'embedding.npz'))


	def save(self, savedir):
		self.token_map.save(os.path.join(savedir, 'token_map.gz'))
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



import time
from multiprocessing import Queue, Process, Pipe
from Queue import Empty
from corpus_reader import CorpusReader, default_parse
from counter_sampler import MultinomialSampler
from token_map import UNK
from unigram_dictionary import UnigramDictionary, WARN, SILENT
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
		unigram_dictionary=None,
		noise_ratio=15,
		kernel=[1,2,3,4,5,5,4,3,2,1],
		t = 1.0e-5,
		batch_size = 1000,
		parse=default_parse,
		verbose=True
	):

		# Get a corpus reader
		self.corpus_reader = CorpusReader(
			files=files, directories=directories, skip=skip, parse=parse,
			verbose=verbose
		)

		# Load the unigram_dictionary
		if unigram_dictionary is not None:
			self.unigram_dictionary = unigram_dictionary
		else:
			self.unigram_dictionary = UnigramDictionary()

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
		existing (pre-filled) UnigramDictionary was loaded, since otherwise 
		it would just return 0.
		'''
		# Delegate to the underlying UnigramDictionary
		return len(self.unigram_dictionary)


	def load(self, directory):
		'''
		Load the unigram_dictionary whose files are stored in <directory>.
		'''
		# Delegate to the underlying UnigramDictionary
		self.unigram_dictionary.load(directory)

	
	def save(self, directory):
		'''
		Save the unigram_dictionary to <directory>.
		'''
		# Delegate to the underlying UnigramDictionary
		self.unigram_dictionary.save(directory)


	def check_access(self, savedir):

		savedir = os.path.abspath(savedir)
		path, dirname = os.path.split(savedir)

		# Make sure that the directory we want exists (make it if not)
		if not os.path.isdir(path):
			raise IOError('%s is not a directory or does not exist' % path)
		if not os.path.exists(savedir):
			os.mkdir(os.path)
		elif os.path.isfile(savedir):
			raise IOError('%s is a file. % savedir')

		# Make sure we can write to the file
		f = open(os.path.join(savedir, '.__test-w2v-access'), 'w')
		f.write('test')
		f.close
		os.remove(os.path.join(savedir, '.__test-w2v-access'))


	def prepare(self, savedir=None, min_frequency=None):
		'''
		Iterate over the entire corpus in order to build a 
		UnigramDictionary.  We need this because we need to sample
		from the unigram distribution in producing minibatches.
		Optionally prune all tokens that occur fewer than min_frequency
		times from dictionary.  Use min_frequency=None (the default) to
		specify no pruning.  Optionally save the dictionary to savedir 
		(this is done after pruning if pruning is requested).
		'''

		# Before doing anything, if we were requested to save the 
		# dictionary, make sure we'll be able to do that (fail fast)
		if savedir is not None:
			self.check_access(savedir)

		# Read through the corpus, building the UnigramDictionary
		for line in self.corpus_reader.read_no_q():
			self.unigram_dictionary.update(line)

		# Prune the dictionary, if requested to do so.
		if min_frequency is not None:
			self.unigram_dictionary.prune(min_frequency)

		# Save the dictionary, if requested to do so.
		if savedir is not None:
			self.save(savedir)


	def prune(self, min_frequency=5):
		'''
		Exposes the prune function for the underlying UnigramDictionary
		'''
		self.unigram_dictionary.prune(min_frequency)
  

	def __iter__(self):

		# Once iter is called, a subprocess will be started which
		# begins generating minibatches.  These accumulate in a queue
		# and iteration pulls from that queue.  That way, iteration
		# can begin as soon as the first minibatch is prepared, and 
		# later minibatches are prepared in the background while earlier
		# minibatches are used.  The idea is that this will keep the 
		# CPU(s) busy while training occurs on the GPU

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
			target=self.enqueue_minibatches,
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
			(self.batch_size, 2), UNK, dtype='int32'
		)
		noise_batch = np.full(
			(self.batch_size * self.noise_ratio, 2), UNK, dtype='int32'
		)
		return signal_batch, noise_batch


	def generate(self):

		chooser = TokenChooser(K=len(self.kernel)/2, kernel=self.kernel)
		signal_batch, noise_batch = self.init_batch()

		# i keeps track of position in the signal batch
		i = -1
		for line in self.corpus_reader.read_no_q():

			# Isolated tokens (e.g. one-word sentences) have no context
			# and can't be used for training.
			if len(line) < 2:
				continue

			token_ids = self.unigram_dictionary.get_ids(line)

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
				noise_context_ids = self.unigram_dictionary.sample(
					(self.noise_ratio,))

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
					yield (signal_batch, noise_batch)
					signal_batch, noise_batch = self.init_batch()
					i = -1

		# Normally we'll have a partially filled minibatch after processing
		# the corpus.  The elements in the batch that weren't overwritten
		# contain UNK tokens, which act as padding.  Enqueue the partial
		# minibatch.
		if i >= 0:
			yield (signal_batch, noise_batch)


	def get_minibatches(self):
		'''
		Reads through the entire corpus, generating all of the minibatches
		up front, storing them in memory as a list.  Returns the list of
		minibatches.
		'''
		minibatches = []
		for signal_batch, noise_batch in self.generate():
			minibatches.append((signal_batch, noise_batch))

		return minibatches


	def enqueue_minibatches(self, minibatch_queue, send_pipe):

		'''
		Reads through the minibatches, placing them on a queue as they
		are ready.  This usually shouldn't be called directly, but 
		is used when the MinibatchGenerator is treated as an iterator, e.g.:

			for signal, noise in my_minibatch_generator:
				do_something_with(signal, noise)

		It causes the minibatches to be prepared in a separate process
		using this function, placing them on a queue, while a generator
		construct pulls them off the queue as the client process requests
		them.  This keeps minibatch preparation running in the background
		while the client process is busy processing previously yielded 
		minibatches.
		'''

		# Continuously iterate through the dataset, enqueing each
		# minibatch.  The consumer will process minibatches from
		# the queue at it's own pace.
		for signal_batch, noise_batch in self.generate():
			minibatch_queue.put((signal_batch, noise_batch))

		# Notify parent process that iteration through the corpus is
		# complete (so it doesn't need to wait for more minibatches)
		send_pipe.send(self.DONE)


	def do_discard(self, token_id):
		'''
		This function helps with downsampling of very common words.
		Returns true when the token should be discarded as a query word
		'''
		probability = self.unigram_dictionary.get_probability(token_id)
		discard_probability = 1 - np.sqrt(self.t/probability)
		do_discard = np.random.uniform() < discard_probability

		#if do_discard:
		#	print 'discarding', self.unigram_dictionary.get_token(token_id)

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


