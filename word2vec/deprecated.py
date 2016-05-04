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


