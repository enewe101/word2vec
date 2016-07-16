from iterable_queue import IterableQueue
from multiprocessing import Process
from categorical import Categorical
from token_map import UNK
from unigram_dictionary import UnigramDictionary
import numpy as np
from theano import shared
import os
import sys


def default_parse(filename):
	tokenized_sentences = []
	for line in open(filename):
		tokenized_sentences.append(line.strip().split())
	return tokenized_sentences


MAX_NUMPY_SEED = 4294967295
def reseed():
	'''
	Makes a hop in the random chain.

	If called before spawning a child processes, it will ensure each child
	generates different random numbers.  Unlike seeding child randomness
	from an os source of randomness, this will produce globally consistent
	results when the parent starts with the same random seed.

	In other words, child processes will all have different random chains
	but the *particular* random chains they have is deterministic and
	reproducible when the parent process randomness is seeded to the same
	value.  So child processes have samples that are independent from one
	another, but reproducible by controlling the seed of the parent process.
	'''
	np.random.seed(np.random.randint(MAX_NUMPY_SEED))


class Minibatcher(object):

	def __init__(
		self,
		files=[],
		directories=[],
		skip=[],
		batch_size = 1000,
		num_example_generators=3,
		verbose=True
	):

		# Register parameters to instance namespace
		self.files = files
		self.directories = directories
		self.skip = skip
		self.batch_size = batch_size
		self.num_example_generators = num_example_generators
		self.verbose = verbose

		self.setup_symbolic_minibatching()

	def check_access(self, savedir):

		savedir = os.path.abspath(savedir)
		path, dirname = os.path.split(savedir)

		# Make sure that the directory we want exists (make it if not)
		if not os.path.isdir(path):
			raise IOError('%s is not a directory or does not exist' % path)
		if not os.path.exists(savedir):
			os.mkdir(savedir)
		elif os.path.isfile(savedir):
			raise IOError('%s is a file. % savedir')

		# Make sure we can write to the file
		f = open(os.path.join(
			savedir, '.__test-minibatch-generator-access'
		), 'w')
		f.write('test')
		f.close
		os.remove(os.path.join(
			savedir, '.__test-minibatch-generator-access'
		))


	def preparation(self):
		'''
		Used to perform any preparation steps that are needed before
		minibatching can be done (not always necesary).  E.g. assembling a
		dictionary that maps tokens to integers, and determining the total
		vocabulary size of the corpus. Or normalizing variables based on
		their total variation throughout the corpus.

		Usually not called directly, but rather, `prepare()` is called,
		and delegates to this method.  Prepare also enables saving data
		generated during this step.  To do so, override the `save()` method
		(which is a no-op by default).
		'''
		pass


	def save(self, dirname):
		'''
		This can be overridden and used to save data generated within
		`preparation()`.  By default this is a no-op.

		INPUTS
		* directory [str]: path to a directory in which to place files
			to be saved.  Will be created if does not exist (containing
			directory must exist).
		'''
		pass


	def prepare(self, savedir=None, *args, **kwargs):
		'''
		Used to perform any preparation steps that are needed before
		minibatching can be done.  E.g. assembling a dictionary that
		maps tokens to integers, and determining the total vocabulary size
		of the corpus.  It is assumed that files will need
		to be saved as part of this process, and that they should be
		saved under `savedir`, with `self.save()` managing the details
		of writing files under `savedir`.

		INPUTS

		* Note About Inputs *
		the call signature of this method is variable and is
		determined by the call signature of the core
		`self.preparation()` method.  Refer to that method's call
		signature.  Minimally, this method accepts `savedir`

		* savedir [str]: path to directory in which preparation files
			should be saved.

		RETURNS
		* [None]
		'''

		# Before doing anything, if we were requested to save the
		# dictionary, make sure we'll be able to do that (fail fast)
		if savedir is not None:
			self.check_access(savedir)

		self.preparation(savedir, *args, **kwargs)

		# Save the dictionary, if requested to do so.
		if savedir is not None:
			self.save(savedir)


	def generate_filenames(self):

		'''
		Generator that yields all filenames (absolute paths) making up the
		dataset.  (Files are specified to the Minibatcher constructor
		files and / or directories.  All listed files and all files directly
		contained in listed directories will be processed, unless they
		match regex patterns in the optional `skip` list.

		(no INPUTS)

		YIELDS
		* [str]: absolute path to a dataset file
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
					print '\tprocessing', filename

				yield filename

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
						print '\tprocessing', filename

					yield filename


	def parse(self, filename):
		'''
		Parses input corpus files into a file-format-independent in-memory
		representation.  Since training examples are read from file,
		this function provides file-reading and parsing functionality,
		and should bring the data stored in corpus files into a form that
		makes generating actual training examples easy.  This function,
		however, should not actually generate training examples; this
		should be done in `build_examples()`.  Rather, this provides a
		modular component that can be exchaged in cases where the file
		format changes, making the example-building logic independent of
		file format logic

		INPUTS
		* filename [str]: path to a corpus file to be read

		RETURNS
		* [any]: file-format-independent representation of training data.
		'''

		# This example parser simply returns a list of all the lines in
		# the file in which each line is tokenized.  Typically this
		# function will need to do more, such as loading json, numpy, or
		# image data file formats.  The return value will be provided to
		# the method `build_examples`.
		return default_parse(filename)


	def build_examples(self, parsed):
		'''
		Builds and yields training examples based on parsed data obtained
		as output from `parse()`.  This should contain all of the logic
		for building training examples based on data from dataset files,
		but should be independent of the dataset storage format (parsing
		the storage format should be handled in `parse()`.

		INPUTS
		* parsed [any]: Data from a single dataset file that has been
			parsed into a file-format-independent form.

		YIELDS
		* [any]: Usually a single training example.  However, depending
			on how you implement `batch_examples()`, this may not be
			necessary.
		'''
		# This minimalistic example assumes that parsed has yielded an
		# iterable of items, and that each item is already a well-formed
		# training example, which can directly be yielded.  Normally
		# substantial transformations and sub-sampling might be done
		# here to generate training examples using the dataset.
		for item in parsed:
			example = item
			yield example


	def process_file(self, filename):
		'''
		Generator that encapsulates all processing for a single data file,
		i.e., parsing and building of examples.

		INPUTS
		* filename [str]: path to file to be processed.

		YIELDS
		* example [any]: object representing (usually) one training
			example.
		'''

		self.cur_file = filename
		parsed = self.parse(filename)
		examples = self.build_examples(parsed)
		for example in examples:
			yield example


	def generate_examples(self):
		'''
		Generates training examples.  Via subroutienes, this iterates
		over all data files, parsing them, and then generating
		training examples from them, which are yielded one at a time.

		(no INPUTS)

		YIELDS
		* [any]: (Usually) single training example.
		'''

		for filename in self.generate_filenames():
			for example in self.process_file(filename):
				yield example


	def generate_examples_async(self, file_queue, example_queue):
		'''
		Asynchronous version of `generate_examples()` designed to be
		run within a forked worker process.  Draws filenames of dataset
		files from a queue, and yields training examples onto a queue.
		This is part of the minibatching pipeline used for asynchronous
		minibatching, which is invoked when Minibatcher is treated
		as an iterable.

		INPUTS
		* file_queue [IterableQueue.ConsumerQueue]: A queue containing
			absolute paths ([str]s) of dataset files.
		* example_queue [IterableQueue.ProducerQueue]: A queue that accepts
			training examples ([any]).

		(no OUTPUS)

		CONSUMES
		* filename [str]: Absolute path to a dataset file.

		PRODUCES
		* example [any]: A single training example
		'''
		for filename in file_queue:
			for example in self.process_file(filename):
				example_queue.put(example)

		example_queue.close()


	def batch_examples(self, example_iterator):
		'''
		Accumulates examples from an example_iterator bundling them into
		minibatches, which are yielded.

		INPUTS
		* example_iterator [iterable]: An iterable that yields examples
			([any]).

		YIEDLS
		* minibatches [iterable]: An iterable that yields examples ([any]).
		'''
		minibatch = []
		for example in example_iterator:
			batch.append(example)
			if len(minibatch) == self.batch_size:
				yield minibatch
				minibatch = []
		if len(minibatch) > 0:
			yield minibatch


	def generate_minibatches(self):
		'''
		Full-pass non-multiprocessing generator of minibatches.
		Iterates over all datafiles, parses them, generates training
		examples, and batches those training examples into minibatches,
		which are yielded.  Can be used as an alternative to `__iter__()`.
		Minibatches will not be prepared in the background (normally
		preparing in the background offers speed advantages).  Instead
		minibatches will be constructed on demand according to the
		pace of looping by the caller.  Normally much slower, but possibly
		desireable if you don't want background batching processes to be
		spawned.

		(no INPUTS)

		YIELDS
		* minibatches [iterable]: Iterable of training examples ([any]).
		'''
		for minibatch in self.batch_examples(self.generate_examples()):
			yield minibatch


	def get_minibatches(self):
		'''
		Full-pass non-multiprocessing generator for minibatches.  Differs
		from `generate_minibatches()` in that it assembles and returns
		an ordinary python list containing the entire training set.
		This can be useful if you want to load the full training set
		into memory for some reason.  However, this is generally slower
		than `__iter__()` because batching does not take place in the
		background using spawned processes, but rather takes place in
		the foreground, blocking until the entire training set has been
		read into memory.  This will usually have similar training speed
		characteristics to `generate_minibatches()`, but will block for
		a long time at the start while data is read into memory, and then
		will catch up as training takes place without the need for further
		reads from storage. Large datasets will take long to load and will
		consume large amounts of memory.

		(no INPUTS)

		OUTPUTS
		* [list : [iterable]]: A list of minibatches.
		'''
		minibatches = []
		for minibatch in self.generate_minibatches():
			minibatches.append(minibatch)

		return minibatches


	def generate_minibatches_async(self, example_queue, minibatch_queue):
		'''
		Asynchronous version of `generate_minibatches()`, designed to be
		run within a forked worker process.  Used in conjuction with
		`generate_examples_async()`, via the sharing of an example_queue,
		to create an asynchronous pipeline that assembles minibatches in
		the background.  This is the default method used for minibatching
		when Minibatcher is treated as an iterator.  See
		`__iter__()`.

		INPUTS
		* example_queue [IterableQueue.ConsumerQueue]: A queue that yields
			training examples ([any]).

		(no OUTPUS)

		CONSUMES
		* [any]: Training examples.

		PRODUCES
		* [iterable]: Minibatches of training examples.
		'''
		for minibatch in self.batch_examples(example_queue):
			minibatch_queue.put(minibatch)
		minibatch_queue.close()


	def get_async_batch_iterator(self):
		'''
		Builds an asynchronous minibatching pipeline, which reads all
		dataset files, parses them, generates training examples, and
		packages those training examples into minibatches.  Finally,
		it yields an iterable of minibatches, taking the form of an
		IterableQueue.ConsumerQueue.

		(no Inputs)

		OUTPUTS
		* [iterable (IterableQueue.ConsumerQueue)]: Iterable of
			minibatches.
		'''

		# TODO: currently the only randomness in minibatching comes from
		# the signal context and noise contexts that are drawn for a
		# given entity query tuple.  But the entity query tuples are read
		# deterministically in order through the corpus  Ideally examples
		# should be totally shuffled..


		file_queue = IterableQueue()
		example_queue = IterableQueue()
		minibatch_queue = IterableQueue()

		# Fill the file queue
		file_producer = file_queue.get_producer()
		for filename in self.generate_filenames():
			file_producer.put(filename)
		file_producer.close()

		# Make processes that process the files and put examples onto
		# the example queue
		for i in range(self.num_example_generators):

			# These calls to np.random are a hack to ensure that each
			# child example-generating process gets different randomness
			#reseed()
			Process(target=self.generate_examples_async, args=(
				file_queue.get_consumer(),
				example_queue.get_producer()
			)).start()

		# Make a processes that batches the files and puts examples onto
		# the minibatch queue
		Process(target=self.generate_minibatches_async, args=(
			example_queue.get_consumer(),
			minibatch_queue.get_producer()
		)).start()

		# Before closing the queues, make a consumer that will be used for
		# yielding minibatches to the external call for iteration.
		self.minibatch_consumer = minibatch_queue.get_consumer()

		# Close all queues
		file_queue.close()
		example_queue.close()
		minibatch_queue.close()

		# Return the minibatch_consumer as the iterator
		return self.minibatch_consumer


	def setup_symbolic_minibatching(self):

		# Create a shared variable that will (later) hold the entire dataset
		self.loaded_dataset = shared(np.array([],dtype='float32'))

		# Define a symbolic variable to represent a single minibatch,
		# based on a slice of the loaded dataset that is indexed by another
		# shared variable
		i = shared(0)
		self.X = self.loaded_dataset[i : i+batch_size,]

		# Define updates that causes the minibatch window to move
		self.updates = [(i, i+batch_size)]


	def get_symbolic_minibatch(self):
		# Return the symbolic minibatch and the updates
		return self.X, self.updates


	def get_symbolic_minibatch(self):
		'''
		This generates a theano shared variable storing the full dataset
		-- all training examples.  When the theano device setting is the
		GPU, shared variables are stored on the GPU, so this has the
		effect of loading the full dataset onto the GPU.

		One of the return values is a (set of) symbolic theano variable(s)
		corresponding to a single minibatch of the data.  This symbolic
		variable can be used to set up the training function.  What will
		happen during training is that this variable acts as a sliding
		"window" on the full dataset, selecting each minibatch in turn,
		even though the entire dataset is loaded into GPU memory.

		The indexing that causes the symbolic minibatch to address different
		parts of the dataset is itself a shared variable, and it can be
		updated using an update tuple provided to the updates list of a
		theano function.  The necessary update tuple is also provided as
		a return value, so that it can be incorporated into the training
		function
		'''

		# Load the entire dataset into RAM.
		# Use the async_batch_iterator to allow multiple read processes.
		dataset = []
		num_batches = 0
		for minibatch in self.get_async_batch_iterator():
			dataset.extend(minibatch)
			num_batches += 1
		dataset = np.array(dataset, dtype="float32")

		# Now move the dataset onto GPU (into GRAM).
		self.loaded_dataset.set_value(dataset)

		# Return the symbolic minibatch and the updates
		return num_batches


	def __iter__(self):
		return self.get_async_batch_iterator()


class NoiseSymbolicMinibatcherMixin(object):

	def setup_symbolic_minibatching(self):

		# Define a shared variable that will (later) hold entire dataset
		# Note that in this case we want it to hold integers, not floats
		# because the word2vec data are integers representing token ids.
		#
		# Here we are just creating an empty shared variable, but we
		# ensure that it has the right number of dimensions (2, because of
		# the [[]] passed into np.array), and correct dtype.
		self.loaded_dataset = shared(np.array([[]],dtype='int32'))

		# Define a the minibatch window indexing around a shared
		# variable, i, which tracks minibatch iteration
		i = shared(0)
		window_size = self.batch_size * (self.noise_ratio + 1)
		signal_start = i * window_size
		signal_end = signal_start + self.batch_size
		noise_start = signal_end
		noise_end = noise_start + self.batch_size * self.noise_ratio

		# Define a symbolic minibatch in terms of the full dataset
		# using the window indexing
		self.symbolic_signal_batch = self.loaded_dataset[
			signal_start : signal_end,]
		self.symbolic_noise_batch = self.loaded_dataset[
			noise_start : noise_end,]

		# Define an update that increments the minibatch
		self.updates = [(i, i+1)]


	def get_symbolic_minibatch(self):
		return (
			self.symbolic_signal_batch, self.symbolic_noise_batch,
			self.updates
		)


	def load_dataset(self):
		'''
		This generates a theano shared variable storing the full dataset
		-- all training examples.  When the theano device setting is the
		GPU, shared variables are stored on the GPU, so this has the
		effect of loading the full dataset onto the GPU.

		One of the return values is a (set of) symbolic theano variable(s)
		corresponding to a single minibatch of the data.  This symbolic
		variable can be used to set up the training function.  What will
		happen during training is that this variable acts as a sliding
		"window" on the full dataset, selecting each minibatch in turn,
		even though the entire dataset is loaded into GPU memory.

		The indexing that causes the symbolic minibatch to address different
		parts of the dataset is itself a shared variable, and it can be
		updated using an update tuple provided to the updates list of a
		theanod function.  The necessary update tuple is also provided as
		a return value, so that it can be incorporated into the training
		function
		'''

		# Load the entire dataset into RAM, converting to a numpy array.
		# Use the async_batch_iterator to allow multiple read processes.
		dataset = []
		num_batches = 0
		for signal_batch, noise_batch in self.get_async_batch_iterator():
			dataset.extend(signal_batch)
			dataset.extend(noise_batch)
			num_batches += 1
		dataset = np.array(dataset, dtype='int32')

		# Now move the dataset onto GPU
		self.loaded_dataset.set_value(dataset)

		# Return the number of batches
		return num_batches



class Word2VecMinibatcher(NoiseSymbolicMinibatcherMixin, Minibatcher):

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
		verbose=True,
		num_example_generators=3
	):

		# Need to register noise ratio before call to super, because
		# super will call an overridden method that expects noise ratio
		# to be set
		self.noise_ratio = noise_ratio

		super(Word2VecMinibatcher, self).__init__(
			files,
			directories,
			skip,
			batch_size,
			num_example_generators,
			verbose
		)

		# Register parameters not already registered by
		# `super().__init__().`
		self.kernel = kernel
		self.t = t

		# Load the unigram_dictionary if any, or construct a new empty one.
		if unigram_dictionary is not None:
			self.unigram_dictionary = unigram_dictionary
		else:
			self.unigram_dictionary = UnigramDictionary()

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
		after Minibatcher.prepare() has been called, or if an
		existing (pre-filled) UnigramDictionary was loaded, since otherwise
		it would just return 0.

		(no INPUTS)

		OUTPUTS
		* [int]: size of vocabulary (including `UNK`).

		'''
		# Delegate to the underlying UnigramDictionary
		return len(self.unigram_dictionary)


	def load(self, directory):
		'''
		Loads the unigram_dictionary from files stored in the supplied
		directory.

		INPUTS
		* directory [str]: Path to a directory in which unigram_dictionary
			files are stored.  Unigram dictionary will look for default
			filenames within that directory.

		OUTPUTS
		* [None]
		'''
		# Delegate to the underlying UnigramDictionary
		self.unigram_dictionary.load(directory)


	def save(self, directory):
		'''
		Save the unigram_dictionary to the given directory.
		'''
		# Delegate to the underlying UnigramDictionary
		self.unigram_dictionary.save(directory)


	def preparation(self, savedir, min_frequency=None):
		# Read through the corpus, building the UnigramDictionary
		for filename in self.generate_filenames():
			for tokens in self.parse(filename):
				self.unigram_dictionary.update(tokens)

		# Prune the dictionary, if requested to do so.
		if min_frequency is not None:
			self.unigram_dictionary.prune(min_frequency)


	def prune(self, min_frequency=5):
		'''
		Exposes the prune function for the underlying UnigramDictionary
		'''
		self.unigram_dictionary.prune(min_frequency)


	def batch_examples(self, example_iterator):

		signal_batch, noise_batch = self.init_batch()

		# i keeps track of position in the signal batch
		i = -1
		for signal_example, noise_examples in example_iterator:

			# Increment position within the batch
			i += 1

			# Add the signal example
			signal_batch[i, :] = signal_example

			# Figure out the position within the noise batch
			j = i*self.noise_ratio

			# block-assign the noise samples to the noise batch array
			noise_batch[j:j+self.noise_ratio, :] = noise_examples

			# Once we've finished assembling a minibatch, enqueue it
			# and start assembling a new minibatch
			if i == self.batch_size - 1:
				yield (signal_batch, noise_batch)
				signal_batch, noise_batch = self.init_batch()
				i = -1

		# Normally we'll have a partially filled minibatch after processing
		# the corpus.  The elements in the batch that weren't overwritten
		# contain UNK tokens, which act as padding.  Yield the partial
		# minibatch.
		if i >= 0:
			yield (signal_batch, noise_batch)



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


	def build_examples(self, parsed):

		chooser = TokenChooser(K=len(self.kernel)/2, kernel=self.kernel)

		for tokens in parsed:

			# Isolated tokens (e.g. one-word sentences) have no context
			# and can't be used for training.
			if len(tokens) < 2:
				continue

			token_ids = self.unigram_dictionary.get_ids(tokens)

			# We'll now generate generate signal examples and noise
			# examples for training
			for query_token_pos, query_token_id in enumerate(token_ids):

				# Possibly discard the token
				if self.do_discard(query_token_id):
					continue

				# Sample a token from the context
				context_token_pos = chooser.choose_token(
					query_token_pos, len(token_ids)
				)
				context_token_id = token_ids[context_token_pos]
				signal_example = [query_token_id, context_token_id]

				# Sample tokens from the noise
				noise_context_ids = self.unigram_dictionary.sample(
					(self.noise_ratio,))

				# block-assign the noise samples to the noise batch array
				noise_examples = [
					[query_token_id, noise_context_id]
					for noise_context_id in noise_context_ids
				]

				# Yield the example
				yield (signal_example, noise_examples)


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
				Categorical(trunc_probabilities)
			)

		# Sample from the multinomial sampler for the context of this shape
		outcome_idx = self.samplers[start,stop].sample()

		# Map this into the +/- indexing relative to the query word
		relative_idx = self.indices[outcome_idx + start]

		# And then map this into absolute indexing
		result_idx = relative_idx + idx

		return result_idx
