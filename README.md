# Theano-word2vec

## Quickstart

The simplest way to train a word2vec embedding:
```python
>>> from word2vec import word2vec
>>> word2vec_embedder, dictionary = word2vec(files=['corpus/file1.txt', 'corpus/file2.txt'])
```
Where the input files should be formatted with one sentence per line, with
tokens space-separated.

Once trained, the embedder can be used to convert words to vectors:
```python
>>> tokens = 'A sentence to embed'.split()
>>> token_ids = dictionary.get_ids(tokens)
>>> vectors = word2vec_embedder.embed(token_ids)
```

The `word2vec()` function exposes most of the basic parameters appearing
in Mikolov's skip-gram model based on noise contrastive estimation:
```python
>>> embedder, dictionary = word2vec(
	savedir='data/my-embedding',		# directory in which to save embedding parameters (deepest directory will be created if it doesn't exist)
	files=['corpus/file1.txt', corpus/file2.txt'],	# List of files comprising the corpus
	directories=['corpus', 'corpus/additional']	# List of directories whose directly-contained files comprise the corpus (you may combine files and directories)
	skip=[re.compile('*.bk$'), re.compile('exclude-from-corpus')],	# matching files and directories will be excluded
	num_epochs=5,				# Number of passes through training corpus
	unigram_dictionary=preexisting_dictionary,	# Specify the dictionary (will be created if not supplied; maps words to ids and tracks word frequencies)
	noise_ratio=15,				# Number of "noise" examples for every signal example
	kernel=[1,2,3,3,2,1],		# Relative probability of sampling context words, assumed symettrically surrounding query word
	t=1.0e-5,				# Threshold used in calculating discard-probability for very common words
	batch_size = 1000		# Size of minibatches during training
	num_embedding_dimensions = 500, # Dimensionality of the embedding vector space 
	word_embedding_init=lasagne.init.Normal(),	# Initializer for embedding parameters (can be a numpy array too)
	context_embedding_init=lasagne.init.Normal(),	# Initializer for context embedding parameters (can be numpy array)
	learning_rate = 0.1,	# Size of stochastic gradient descent steps during training
	momentum=0.9,		# Amount of Nesterov momentum during training
	verbose=True		# Print messages during training
	num_example_generators=3	# Number of parrallel corpus-reading processes used to generate minibatches
)
```

For more customization, check out the documentation (soon) to see how to 
assemble your own training setup using the classes provided in word2vec.
