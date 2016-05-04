# Theano-word2vec

## Quickstart

The simplest way to train a word2vec embedding:
```python
>>> from word2vec import word2vec
>>> embedder, dictionary = word2vec(files=['corpus/file1.txt', 'corpus/file2.txt'])
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
...		# directory in which to save embedding parameters (deepest directory will be created if it doesn't exist)
...		savedir='data/my-embedding',
...		# List of files comprising the corpus
...		files=['corpus/file1.txt', 'corpus/file2.txt'],	
...		# Include whole directories of files (deep files not included)
...		directories=['corpus', 'corpus/additional'],
...		# Indicate files to exclude using regexes
...		skip=[re.compile('*.bk$'), re.compile('exclude-from-corpus')],	
...		# Number of passes through training corpus
...		num_epochs=5,				
...		# Specify the mapping from tokens to ints (else create it automatically)
...		unigram_dictionary=preexisting_dictionary,	
...		# Number of "noise" examples included for every "signal" example
...		noise_ratio=15,	
...		# Relative probability of skip-gram sampling centered on query word
...		kernel=[1,2,3,3,2,1],		
...		# Threshold used to calculate discard-probability for query words
...		t=1.0e-5,				
...		# Size of minibatches during training
...		batch_size = 1000,
...		# Dimensionality of the embedding vector space 
...		num_embedding_dimensions = 500, 
...		# Initializer for embedding parameters (can be a numpy array too)
...		word_embedding_init=lasagne.init.Normal(),	
...		# Initializer for context embedding parameters (can be numpy array)
...		context_embedding_init=lasagne.init.Normal(),	
...		# Size of stochastic gradient descent steps during training
...		learning_rate = 0.1,	
...		# Amount of Nesterov momentum during training
...		momentum=0.9,		
...		# Print messages during training
...		verbose=True,
...		# Number of parrallel corpus-reading processes 
...		num_example_generators=3	
)
```

For more customization, check out the documentation (soon) to see how to 
assemble your own training setup using the classes provided in word2vec.
