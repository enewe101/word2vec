Theano-word2vec
~~~

::
    >>> from word2vec import Word2Vec

    >>> word2vec = Word2Vec()

    >>> # Train an embedding on a corpus
    >>> word2vec.train_on_corpus(
    ...     open('my-corpus.txt').read(),
    ...     num_embedding_dimensions=500
    ... )

    >>> # Get embeddings using a trained model
    >>> embeddings = word2vec.embed(
    ...     'this will produce a list of vectors.  If the input is a string it '
    ...     'gets tokenized'
    ... ) 
    >>> other_embeddings = word2vec.embed([
    ...     'control', 'tokenization', 'by', 'passing', 'a', 'tokenized', 'list' 
    ... ])

    >>> # Do anological arithmetic
    >>> king, man, woman = word2vec.embed('king man woman') 
    >>> queen = king - man + woman
    >>> print word2vec.nearest(queen)
    'queen'

    >>> # Save and load models
    >>> word2vec.save('my-embedding.npz')
    >>> word2vec.load('my-embedding.npz')

    >>> # Get a fresh Lasagne layer out of the trained model
    >>> my_deep_learning_architecture = word2vec.layer(input_layer)


