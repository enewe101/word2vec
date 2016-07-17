import re
import gc
import time
from iterable_queue import IterableQueue
from multiprocessing import Process
from subprocess import check_output
from categorical import Categorical
from token_map import UNK
from unigram_dictionary import UnigramDictionary
import numpy as np
from theano import shared
import os
import sys


class DataSetReaderIllegalStateException(Exception):
	'''
	Used if DatasetReader's methods are called in an incorrect order, e.g.
	calling generate_dataset() before calling prepare() on a DatasetReader
	that was not initialized with a UnigramDictionary.
	'''
	pass


class DatasetReader(object):

	def __init__(
		self,
		files=[],
		directories=[],
		skip=[],
		batch_size = 1000,
		noise_ratio=15,
		t=1e-5,
		num_processes=3,
		unigram_dictionary=None,
		kernel=[1,2,3,4,5,5,4,3,2,1],
		verbose=True
	):

		# Register parameters to instance namespace
		self.files = files
		self.directories = directories
		self.skip = skip
		self.batch_size = batch_size
		self.t = t
		self.noise_ratio = noise_ratio
		self.num_processes = num_processes
		self.unigram_dictionary = unigram_dictionary
		self.kernel = kernel
		self.verbose = verbose

		# If unigram dictionary not supplied, make one
		if unigram_dictionary is None:
			self.prepared = False
			self.unigram_dictionary = UnigramDictionary()

		self.data_loaded = False


	def check_access(self, savedir):
		'''
		Test out writing in savedir.  The processes that generate the data
		to be saved can be really long-running, so we want to find out if there
		is a simple IOError early!
		'''
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

		# Process the files listed in `files`, unles they match an entry in skip
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


	def __iter__(self):
		return self.stream_examples()


	def stream_examples(self):
		'''
		Reads all dataset files and generates examples. Signal and noise
		examples are consistently distributed in alternating chunks of
		batch_size signal examples followed by batch_size noise_examples.
		Padding with null examples is used at the end of the dataset.

		(no Inputs)
		Yields
			* example [any]
		'''

		# TODO: currently the only randomness in minibatching comes from
		# the signal context and noise contexts that are drawn for a
		# given entity query tuple.  But the entity query tuples are read
		# deterministically in order through the corpus  Ideally examples
		# should be totally shuffled..

		# This cannot be called before calling prepare(), unless a prepared
		# UnigramDictionary was passed to the self's constructor
		if not self.prepared:
			raise DataSetReaderIllegalStateException(
				"DatasetReader: generate_examples() cannot be called before "
				"prepare() is called unless a prepared UnigramDictionary has "
				"was passed into the DatasetReader's constructor."
			)

		# Open queues, which will help with multiprocessing the work
		file_queue = IterableQueue()
		signal_example_queue = IterableQueue()
		noise_example_queue = IterableQueue()

		# Fill the file queue with all dataset files that need to be processed
		file_producer = file_queue.get_producer()
		for filename in self.generate_filenames():
			file_producer.put(filename)
		file_producer.close()

		# Read dataset files and generate examples using multiple processes
		for i in range(self.num_processes):
			# Give child processes independent (yet reproducible) randomness
			reseed()
			# Start a child on reading dataset files and generating examples
			Process(target=self.generate_examples, args=(
				file_queue.get_consumer(),
				signal_example_queue.get_producer(),
				noise_example_queue.get_producer()
			)).start()

		# Pull together and organize the examples generated
		organized_stream = self.organize_examples(
			signal_example_queue.get_consumer(),
			noise_example_queue.get_consumer()
		)

		# Close all the queues
		file_queue.close()
		signal_example_queue.close()
		noise_example_queue.close()

		# Yield an organized stream of results
		for example in organized_stream:
			yield example


	def produce_examples(self, filename_iterator):

		sorted_examples = []
		for filename in filename_iterator:

			# Parse the file, then generate a bunch of examples from it
			parsed = self.parse(filename)
			examples = self.build_examples(parsed)

			signal_examples = []
			noise_examples = []

			some_signal_examples_remain = True
			while some_signal_examples_remain:

				try:
					while len(signal_examples) < self.batch_size:
						is_signal, example = examples.next()
						if is_signal:
							signal_examples.append(example)
						else:
							noise_examples.append(example)

				except StopIteration:
					remaining = self.batch_size - len(signal_examples)
					padding = [self.make_null_example() for i in range(remaining)]
					signal_examples.extend(padding)
					some_signal_examples_remain = False

				sorted_examples.extend(signal_examples[:self.batch_size])
				signal_examples = signal_examples[self.batch_size:]

				try:
					while len(noise_examples) < self.batch_size * self.noise_ratio:
						is_signal, example = examples.next()
						if is_signal:
							signal_examples.append(example)
						else:
							noise_examples.append(example)

				except StopIteration:
					remaining = self.batch_size * self.noise_ratio - len(noise_examples)
					padding = [self.make_null_example() for i in range(remaining)]
					noise_examples.extend(padding)

				sorted_examples.extend(noise_examples[:self.batch_size * self.noise_ratio])
				noise_examples = noise_examples[self.batch_size * self.noise_ratio:]

		# Cast the list to a numpy int32 array.  Keep a reference in self.
		if len(sorted_examples) > 0:
			sorted_examples = np.array(sorted_examples, dtype='int32')
		else:
			sorted_examples = np.empty(shape=(0,2))
		return sorted_examples


	def generate_dataset_serial(self, compiled_dataset_dir):
		'''
		Generate the dataset from files handed to the constructor.  A single
		process is used, and all the data is stored in a single file at
		'compiled_dataset_dir/examples/0.npz'.
		'''

		# We save dataset in the "examples" subdir of the model_dir
		save_dir = os.path.join(compiled_dataset_dir, 'examples')
		if not os.path.exists(save_dir):
			os.mkdir(save_dir)

		# Generate the data for each file
		file_iterator = self.generate_filenames()
		self.examples = self.produce_examples(file_iterator)
		self.data_loaded = True

		# Save the data
		save_path = os.path.join(save_dir, '0.npz')
		print save_path
		np.savez(save_path, data=self.examples)

		# Return it
		return self.examples


	def generate_dataset_worker(self, file_iterator, macrobatch_queue):
		macrobatch = self.produce_examples(file_iterator)
		macrobatch_queue.put(macrobatch)
		macrobatch_queue.close()


	def generate_dataset_parallel(self, compiled_dataset_dir):
		'''
		Parallel version of generate_dataset_serial.  Each worker is responsible
		for saving its own part of the dataset to disk, called a macrobatch.
		the files are saved at 'compiled_dataset_dir/examples/<batch-num>.npz'.
		'''

		# We save dataset in the "examples" subdir of the model_dir
		save_dir = os.path.join(compiled_dataset_dir, 'examples')
		if not os.path.exists(save_dir):
			os.mkdir(save_dir)

		file_queue = IterableQueue()
		macrobatch_queue = IterableQueue()

		# Put all the filenames on a producer queue
		file_producer = file_queue.get_producer()
		for filename in self.generate_filenames():
			print 'placing on file queue', filename
			file_producer.put(filename)
		file_producer.close()

		# Start a bunch of worker processes
		for process_num in range(self.num_processes):
			args = (
				file_queue.get_consumer(),
				macrobatch_queue.get_producer()
			)
			Process(target=self.generate_dataset_worker, args=args).start()

		# This will receive the macrobatches from all workers
		macrobatch_consumer = macrobatch_queue.get_consumer()

		# Close the iterable queues
		file_queue.close()
		macrobatch_queue.close()

		# Retrieve the macrobatches from the workers, write them to file
		macrobatches = []
		for macrobatch_num, macrobatch in enumerate(macrobatch_consumer):
			save_path = os.path.join(save_dir, '%d.npz' % macrobatch_num)
			np.savez(save_path, data=macrobatch)
			macrobatches.append(macrobatch)

		# Concatenate the macrobatches, and return the dataset
		self.examples = np.concatenate(macrobatches)
		self.data_loaded = True
		return self.examples


	def generate_dataset(self):
		'''
		Reads all examples and stores them in self.examples.  Converts the
		examples to an numpy int32 array.
		(no Inputs)
		(no Outpus)
		Side Effects
			* populates self.examples [np.array (int32)]
		'''
		# Start streaming examples (generated by child processes), capturing
		# them in a big plain Python list
		self.examples = []
		for example in self.stream_examples():
			self.examples.append(example)

		# Cast the list to a numpy int32 array.  Keep a reference in self.
		self.examples = np.array(self.examples, dtype='int32')
		self.data_loaded = True

		# Return the dataset
		return self.examples


	def parse(self, filename):
		'''
		Parses input corpus files into a file-format-independent in-memory
		representation.  The output of this function is passed into
		`build_examples` for any processing that is needed, irrespective of
		file format, to generate examples form the stored data.

		INPUTS
		* filename [str]: path to a corpus file to be read

		RETURNS
		* [any]: file-format-independent representation of training data.
		'''
		tokenized_sentences = []
		for line in open(filename):
			tokenized_sentences.append(line.strip().split())
		return tokenized_sentences


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


	def load_dictionary(self, directory):
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


	def save_data(self, directory):
		'''
		Save the generated dataset to the given directory.
		'''
		if not self.data_loaded:
			raise DataSetReaderIllegalStateException(
				'DatasetReader: cannot save the dataset before any data has '
				'been generated.'
			)
		path = os.path.join(directory, 'data.npz')
		print 'saving to', path
		np.savez(path, data=self.examples)


	MATCH_EXAMPLE_STORE = re.compile(r'[0-9]+\.npz')
	def load_data(self, compiled_dataset_dir):
		'''
		Load the dataset from the given directory
		'''
		examples_dir = os.path.join(compiled_dataset_dir, 'examples')
		fnames = check_output(['ls %s' % examples_dir], shell=True).split()
		macrobatches = []
		for fname in fnames:
			if not self.MATCH_EXAMPLE_STORE.match(fname):
				print 'skipping', fname
				continue
			f = np.load(os.path.join(examples_dir, fname))
			macrobatches.append(f['data'].astype('int32'))

		if len(macrobatches) < 1:
			raise IOError(
				'DatasetReader: no example data files found in %s.' % examples_dir
			)

		self.examples = np.concatenate(macrobatches)
		self.data_loaded = True
		return self.examples


	def save_dictionary(self, directory):
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

		self.prepared = True


	def prune(self, min_frequency=5):
		'''
		Exposes the prune function for the underlying UnigramDictionary
		'''
		self.unigram_dictionary.prune(min_frequency)


	def build_examples(self, parsed):
		'''
		Using the data of a parsed file, generates examples.  Two kinds of
		examples are generated --- signal and noise.  They are yielded in a
		tuple, along with a flag indicating whether the particular example is
		a signal, i.e.: (is_signal, example)
		'''

		chooser = TokenChooser(K=len(self.kernel)/2, kernel=self.kernel)

		i = 0
		j = 0
		for tokens in parsed:

			i += 1
			if i % 500 == 0:
				print i
				print '\t', j

			# Isolated tokens (e.g. one-word sentences) have no context
			# and can't be used for training.
			if len(tokens) < 2:
				continue

			token_ids = self.unigram_dictionary.get_ids(tokens)

#			loop_start = time.time()
#			start = loop_start
			# We'll now generate generate signal examples and noise
			# examples for training
			for query_token_pos, query_token_id in enumerate(token_ids):



				# TODO: handle this differently, as post processing after entire
				# dataset created?
				#
				# Possibly discard the token
				if self.do_discard(query_token_id):
					continue

#				discard_decision = time.time() - start
#				start = time.time()

				# Sample a token from the context
				context_token_pos = chooser.choose_token(
					query_token_pos, len(token_ids)
				)
				context_token_id = token_ids[context_token_pos]
				signal_example = [query_token_id, context_token_id]
				j += 1
				yield (True, signal_example)

#				context_sampling = time.time() - start
#				start = time.time()

				# Sample tokens from the noise
				noise_context_ids = self.unigram_dictionary.sample(
					(self.noise_ratio,))

#				noise_sampling = time.time() - start
#				start = time.time()

				# block-assign the noise samples to the noise batch array
				for noise_context_id in noise_context_ids:
					noise_example = [query_token_id, noise_context_id]
					j += 1
					yield (False, noise_example)

#				assign_noise = time.time() - start
#				total = time.time()

#				j += 1
#				if j % 100 == 0:
#					print
#					print 'discard-decision', discard_decision / total
#					print 'context-sampling', context_sampling / total
#					print 'noise-sampling', noise_sampling / total
#					print 'assign-noise:', assign_noise / total
#					print
#					print '-'*32


				loop_start = time.time()
				start = loop_start

	def parsed(self, filename):
		tokenized_sentences = []
		for line in open(filename):
			tokenized_sentences.append(line.strip().split())
		return tokenized_sentences


	def generate_examples(self, file_queue, signal_example_queue, noise_example_queue):
		'''
		Draw file names from the iterable `file_queue`, parse the file into
		units (examples) that will be fed into a batching process.

		INPUTS
		* file_queue [IterableQueue.ConsumerQueue]: A queue containing
			absolute paths ([str]s) of dataset files.
		* signal_example_queue [IterableQueue.ProducerQueue]: A queue that accepts
			training examples ([any]).
		* noise_example_queue [IterableQueue.ProducerQueue]: A queue that accepts
			training examples ([any]).

		(no OUTPUS)

		CONSUMES
		* filename [str]: Absolute path to a dataset file.

		PRODUCES
		* example [any]: A single training example
		'''

		for filename in file_queue:

			# Parse the file, then generate a bunch of examples from it
			parsed = self.parse(filename)
			examples = self.build_examples(parsed)

			# Put the examples onto a queue.  Keep signal and noise examples
			# separate so that they can be combined in a consistent pattern
			for is_signal, example in examples:
				if is_signal:
					signal_example_queue.put(example)
				else:
					noise_example_queue.put(example)

		signal_example_queue.close()
		noise_example_queue.close()


	def make_null_example(self):
		return [UNK, UNK]


	def organize_examples(
		self, signal_example_queue, noise_example_queue
	):
		'''
		Combine signal and noise examples in fixed proportions.  Ensures that
		each minibatch has a contiguous bunch of signal examples, followed by
		a bunch of noise examples, in a consistent proportion.

		INPUTS
		* signal_example_queue [IterableQueue.ConsumerQueue]: A queue that yields
			training examples ([any]).
		* noise_example_queue [IterableQueue.ConsumerQueue]: A queue that yields
			training examples ([any]).

		(no OUTPUS)

		CONSUMES
		* [any]: Signal examples.
		* [any]: Noise examples.

		Side effect:
		* populates self.examples [list (any)]
		'''
		# Draw examples off the signal and noise queue, and organize them into
		# a stream of examples that consistently alternates between
		# batch_size number of signal examples followed by
		# batch_size * noise_ratio number of noise examples

		examples_left = True
		while examples_left:

			# Put batch_size signal examples in a row (add padding if necessary)
			for i in range(self.batch_size):
				try:
					yield signal_example_queue.next()
				except StopIteration:
					examples_left = False
					yield self.make_null_example()

			# Then put batch_size * noise_ratio noise examples. Pad if needed.
			for i in range(self.batch_size * self.noise_ratio):
				try:
					yield noise_example_queue.next()
				except StopIteration:
					yield self.make_null_example()


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


class TheanoMinibatcher(object):
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

	def __init__(self, batch_size=1000, dtype="float32", num_dims=2):
		self.batch_size = batch_size
		self.dtype = dtype
		self.num_dims = num_dims

		self._setup_batching()


	def _initialize_data_container(self, num_dims, dtype):

		# Validate num_dims
		if(num_dims < 1 or not isinstance(num_dims, int)):
			raise ValueError(
				'TheanoMinibatcher: num_dims must be an integer equal to or '
				'greater than 1.'
			)

		# Create the first dim, which houses the dataset
		data_container = []
		num_dims -= 1

		# Repeatedly add a nested dimension, so that we have num_dims nestings
		nested_container_handle = data_container
		for dim_num in range(num_dims):
			new_inner_container = []
			nested_container_handle.append(new_inner_container)
			nested_container_handle = new_inner_container

		return np.array(data_container, dtype=dtype)

	def reset(self):
		'''
		Reset the internal batch_num pointer to the start of the dataset
		'''
		self.batch_num.set_value(0)


	def _setup_batching(self):

		# Make an empty shared variable that will store the dataset
		# Although empty, we can setup the relationship between the
		# minibatch variable and the full dataset
		self.dataset = shared(
			self._initialize_data_container(self.num_dims, self.dtype)
		)

		# Make minibatch by indexing into the dataset
		self.batch_num = shared(0)
		batch_start = self.batch_num * self.batch_size
		batch_end = batch_start + self.batch_size
		self.batch = self.dataset[batch_start : batch_end,]

		# Define an update that moves the batch window through the dataset
		self.updates = [(self.batch_num, self.batch_num+1)]


	def load_dataset(self, dataset):
		# Load the dataset onto the gpu
		self.dataset.set_value(dataset)
		# Determine the total number of minibatches
		self.num_batches = int(np.ceil(len(dataset) / float(self.batch_size)))
		return self.num_batches


	def get_batch(self):
		return self.batch


	def get_updates(self):
		return self.updates


	def get_num_batches(self):
		return self.num_batches



#class Word2VecMinibatcher(NoiseSymbolicMinibatcherMixin, Minibatcher):

#	NOT_DONE = 0
#	DONE = 1

#	def __init__(
#		self,
#		files=[],
#		directories=[],
#		skip=[],
#		unigram_dictionary=None,
#		noise_ratio=15,
#		kernel=[1,2,3,4,5,5,4,3,2,1],
#		t = 1.0e-5,
#		batch_size = 1000,
#		verbose=True,
#		num_processes=3
#	):

#		# Need to register noise ratio before call to super, because
#		# super will call an overridden method that expects noise ratio
#		# to be set
#		self.noise_ratio = noise_ratio

#		super(Word2VecMinibatcher, self).__init__(
#			files,
#			directories,
#			skip,
#			batch_size,
#			num_processes,
#			verbose
#		)

#		# Register parameters not already registered by
#		# `super().__init__().`
#		self.kernel = kernel
#		self.t = t

#		# Load the unigram_dictionary if any, or construct a new empty one.
#		if unigram_dictionary is not None:
#			self.unigram_dictionary = unigram_dictionary
#		else:
#			self.unigram_dictionary = UnigramDictionary()

#		# Validate the kernel.  It should reflect the relative
#		# frequencies of choosing tokens from a window of +/- K tokens
#		# relative to a query token.  So it must have an even number of
#		# entries
#		if not len(self.kernel) % 2 == 0:
#			raise ValueError(
#				'kernel should reflect the relative frequencies of '
#				'selecting a context token within +/- K of the query '
#				'token, and so should have an equal number of entries '
#				'defining frequencies to the left and right of the query '
#				'token, and so should have an even number of entries.'
#			)




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
