'''
This module keeps track of the vocabulary present in a corpus, provides
a two-way maping between tokens (strings) and token_ids (integers),
keeps track of token frequencies, and can yield samples from the
unigram distribution.
'''

import os
from token_map import TokenMap, SILENT, WARN, ERROR, UNK
from counter_sampler import CounterSampler

class UnigramDictionary(object):
    '''
    Bundles together a TokenMap and CounterSampler.  Provides a method for
    pruning the vocabluary while keeping the TokenMap and CounterSampler
    in sync with one another.
    '''


    def __init__(self, on_unk=WARN, token_map=None, counter_sampler=None):
        '''
        Create a new UnigramDictionary.  Typical usage provides no
        arguments, but a token_map and counter_sampler can be provided
        to build a UnigramDictionary that comprises them.
        '''
        self.on_unk = on_unk
        self.token_map = token_map
        if token_map is None:
            self.token_map = TokenMap(on_unk=on_unk)

        self.counter_sampler = counter_sampler
        if counter_sampler is None:
            self.counter_sampler = CounterSampler()


    def __contains__(self, token):
        return token in self.token_map.map


    def sort(self):
        unk_count = self.counter_sampler.counts[0]

        # Get the counts and tokens (skipping the first UNK entry)
        # They are parallel arrays (ith count corresponds to ith token)
        counts = self.counter_sampler.counts[1:]
        tokens = self.token_map.tokens[1:]

        # Zip them together and sort by counts
        token_counts = zip(counts, tokens)
        token_counts.sort(reverse=True)

        # Separate them again
        new_counts = [unk_count]
        new_tokens = ['UNK']
        for count, token in token_counts:
            new_counts.append(count)
            new_tokens.append(token)

        # Rebuild the token_map and counter_sampler on the sorted arrays
        self.token_map = TokenMap(on_unk=self.on_unk, tokens=new_tokens)
        self.counter_sampler = CounterSampler(counts=new_counts)


    def remove(self, token):
        idx = self.get_id(token)
        self.token_map.remove(token)
        self.counter_sampler.remove(idx)


    def compact(self):
        self.token_map.compact()
        self.counter_sampler.compact()


    def prune(self, min_frequency=5):
        '''
        Remove all tokens that have been observed fewer than min_frequency
        times.  Counts for tokens that are removed are attributed to UNK.
        '''
        counts = []
        tokens = []
        discarded = set()
        for idx, token in enumerate(self.token_map.tokens):

            # Copy over tokens that have at least min_frequency
            # observations. Also copy over UNK no matter what it's
            # frequency.
            if (
                self.counter_sampler.get_frequency(idx) >= min_frequency
                or idx == 0
            ):
                tokens.append(token)
                counts.append(self.get_frequency(idx))

            # Skip tokens that have too little frequency.  Attribute their
            # observations to UNK
            else:
                counts[UNK] += self.get_frequency(idx)
                discarded.add(token)

        # Create a new TokenMap and CounterFrequency based on the
        # filtered tokens and their counts
        self.token_map = TokenMap(on_unk=self.on_unk, tokens=tokens)
        self.counter_sampler = CounterSampler(counts=counts)

        return discarded


    def add(self, token):
        '''
        Add a new token.  If this "token type" (which means this specific
        spelling of a word) has not been seen before, add it to the
        mapping.  Also increment the count for that token type.  Return
        its ID under the token mapping.
        '''

        # Get or create an id for this token
        token_id = self.token_map.add(token)

        # Increment the frequency count
        self.counter_sampler.add(token_id)

        return token_id


    def add_count(self, token, count):
        '''
        Add `count` to the counts for `token`, making a new entry if 
        necessary.
        '''
        # Get or create an id for this token
        token_id = self.token_map.add(token)
        # Increment the frequency count
        self.counter_sampler.add_count(token_id, count)


    def get_vocab_size(self):
        '''
        Return the number of unique tokens in the token_map.
        '''
        return len(self.token_map)


    def get_num_tokens(self):
        '''
        Return the total number of (non-distinct) tokens observed.
        '''
        return len(self.counter_sampler)


    def __len__(self):
        '''
        Same as get_vocab_size().
        Return the number of unique tokens in the token_map.
        '''
        return len(self.token_map)


    def update(self, token_iterable):
        '''
        Like `add`, but accepts an iterable of tokens, incrementing the
        count for each of them.
        '''
        return [self.add(token) for token in token_iterable]


    def add_dictionary(self, other):
        '''
        Adds counts from another UnigramDictionary, `other`, to `self`'s
        counts, i.e. adding in place.
        '''
        self.update_counts(other.get_frequency_list())


    def update_counts(self, token_counts_iterable):
        '''
        Like `add_count` but accepts an iterable of (token,count) pairs,
        and increments the count for each token by the count given.
        Expected usage is to have a dictionary with tokens as keys
        and counts as values, and pass in your_dict.iteritems().
        '''
        return [
            self.add_count(token, count) 
            for token, count in token_counts_iterable
        ]


    def get_id(self, token):
        '''
        Get the id (int) for the corresponding token (string).
        '''
        # Delegate to the underlying token_map.
        return self.token_map.get_id(token)


    def get_ids(self, token_iterable):
        '''
        Get the ids (list of ints) for the corresponding tokens (strings)
        issued by token_iterable.
        '''
        # Delegate to the underlying token map.
        return self.token_map.get_ids(token_iterable)


    def get_token(self, idx):
        '''
        Return token (string) for the corresponding id (int)
        '''
        # Delegate to the underlying token map
        return self.token_map.get_token(idx)


    def get_tokens(self, idx_iterable):
        '''
        Return the tokens (list of strings) for the corresponding ids
        (ints) issued by idx_iterable.
        '''
        # Delegate to the underlying token map.
        return self.token_map.get_tokens(idx_iterable)


    def save(self, savedir):
        '''
        Save the UnigramDictionary to the directory specified.  This saves
        the underlying TokenMap and CounterSampler in the directory
        given (savedir), using the default filenames "token-map.gz" and
        "counter-sampler.gz".
        '''

        # If the directory provided is a file, raise an error
        if os.path.exists(savedir):
            if os.path.isfile(savedir):
                raise IOError(
                    'Directory specified for saving UnigramDictionary is a '
                    'file.'
                )

        # If the directory provided doesn't exist, make it (this will not
        # make parent directories though).
        else:
            os.mkdir(savedir)


        # Save the TokenMap and CounterSampler by delegating to their
        # save functions.
        self.token_map.save(os.path.join(savedir, 'token-map.gz'))
        self.counter_sampler.save(os.path.join(
            savedir, 'counter-sampler.gz'
        ))


    def load(self, loaddir):
        '''
        Load a UnigramDictionary from the specified directory, by
        loading the TokenMap and CounterSampler stored there.  This assumes
        the filenames are 'token-map.gz' and 'counter-sampler.gz'.
        '''
        # Load the TokenMap by delegation to its load function
        self.token_map = TokenMap()
        self.token_map.load(os.path.join(loaddir, 'token-map.gz'))

        # Load the CounterSampler by delegation to its load function
        self.counter_sampler = CounterSampler()
        self.counter_sampler.load(
            os.path.join(loaddir, 'counter-sampler.gz'))


    def get_token_list(self):
        '''
        Gets an iterable of tokens currently in the dictionary.  Omits
        The 'UNK' token.
        '''
        return (
            token for token in self.token_map.tokens if token is not 'UNK'
        )


    def get_frequency_list(self):
        '''
        Gets an iterable of (token, count) tuples.
        '''

        # Handle the case where there are no counts at all yet
        if len(self.counter_sampler.counts) == 0:
            return []

        # Otherwise get the counts normally
        return (
            (token, self.get_frequency(self.get_id(token)))
            for token in self.token_map.tokens
        )


    def sample(self, shape=None):
        '''
        Draw a sample according to the counter_sampler probability
        '''
        # Delegate to the underlying CounterSampler
        return self.counter_sampler.sample(shape)


    def get_probability(self, token_id):
        '''
        Return the probability associated to token_id.
        '''
        # Delegate to the underlying CounterSampler
        return self.counter_sampler.get_probability(token_id)


    def get_token_frequency(self, token):
        '''
        Return the frequency (count) associated to the token
        '''
        token_id = self.get_id(token)
        # If the token is unknown, return 0
        if token_id == UNK:
            return 0
        return self.get_frequency(token_id)


    def get_frequency(self, token_id):
        '''
        Return the frequency associated to token_id.
        '''
        # Delegate to the underlying CounterSampler
        return self.counter_sampler.get_frequency(token_id)

