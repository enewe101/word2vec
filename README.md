Theano-word2vec
---------------

To start word2vec, import and instantiate a Word2Vec object:

    >>> from word2vec import Word2Vec
    >>> word2vec = Word2Vec()

You can train on a corpus by passing it as a string::

    >>> word2vec.train_on_corpus(
    ...     open('my-corpus.txt').read(),
    ...     num_embedding_dimensions=500
    ... )

Convert any string into a set of vectors using the embedding learned 
during training::

    >>> embeddings = word2vec.embed(
    ...     'this will produce a list of vectors.  If the input is a string it '
    ...     'gets tokenized'
    ... ) 

Optionally pre-tokenize the string to be embedded::

    >>> other_embeddings = word2vec.embed([
    ...     'control', 'tokenization', 'by', 'passing', 'a', 'tokenized', 'list' 
    ... ])

Do anological arithmetic, and find the word having the nearest embedding
to a given vector::

    >>> king, man, woman = word2vec.embed('king man woman') 
    >>> queen = king - man + woman
    >>> print word2vec.nearest(queen)
    'queen'

Get a fresh Lasagne layer out of the trained model, and use it seemlessly with other Lasagne layers::

    >>> some_input_var = theano.dmatrix('input')
    >>> input_layer = lasagne.layers.InputLayer(some_input_var, shape)
    >>> embedding_layer = word2vec.layer(input_layer)
    >>> my_cool_architecture = lasagne.layers.DenseLayer(embedding_layer)

