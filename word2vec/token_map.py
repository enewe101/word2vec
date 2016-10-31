'''
The TokenMap provides a mapping from tokens (plain words) to integer ids,
as well as a map from ids back to tokens.
'''

import gzip


SILENT = 0
WARN = 1
ERROR = 2
UNK = 0

def ensure_str(s):
	'''
	Ensures that the string is encoded as str, not unicode
	'''
	try:
		return s.encode('utf8')
	except UnicodeDecodeError:
		return s


# TODO: enable loading and saving
class TokenMap(object):

	def __init__(self, on_unk=WARN, tokens=None):
		'''
		Create a new TokenMap.  Most common usage is to call this without
		any arguments.

		on_unk:	Controls the behavior when asked to provide the token_id
			for a token not found in the map.  Default is WARN, which 
			means returning 0 (which is id reserved for unknown tokens) 
			and then printing a warning to stout.  Choose from SILENT, 
			WARN, or ERROR.

		tokens:	List of strings corresponding to a map that should be 
			used.  The index of a token in the list is used as its ID.
			Not normally used, because TokenMap provides functions to 
			build the map easily from a corpus.  The first element in the 
			list should be 'UNK', becuase id 0 is reserved for unknown 
			tokens.  Not doing so is an error.
		'''

		# Validate on_unk
		if on_unk not in (SILENT, WARN, ERROR):
			raise ValueError(
				'on_unk must be one of token_map.SILENT, token_map.WARN, '
				'or token_map.ERROR.'
			)
		self.on_unk = on_unk

		# Initialize the token mapping
		if tokens is None:
			self.map = {'UNK':UNK}	# keys are tokens, values are ids
			self.tokens = ['UNK']	# entries are tokens, indices are ids

		# If an initial lexicon was provided, build the map from it
		else:
			if tokens[0] != 'UNK':
				raise ValueError(
					'tokens[0] must be "UNK" because ID 0 is reserved for '
					'unknown tokens.'
				)

			self.tokens = [ensure_str(t) for t in tokens]
			self.map = dict((t, idx) for idx, t in enumerate(self.tokens))


	def compact(self):
		'''
		Recreate the tokens list and mapping such that `None`s are 
		removed (which are holes left by calls to `remove()`.
		'''
		self.tokens = [t for t in self.tokens if t is not None]
		self.map = dict((t, idx) for idx, t in enumerate(self.tokens))


	def remove(self, token):
		token = ensure_str(token)
		idx = self.get_id(token)
		if idx == UNK:
			raise ValueError(
				'Cannot remove token %s because it does not ' 
				'exist or is reserved.' % str(token)
			)
		self.tokens[idx] = None


	def add(self, token):
		token = ensure_str(token)
		try:
			return self.map[token]
		except KeyError:
			next_id = len(self.tokens)
			self.map[token] = next_id
			self.tokens.append(token)
			return next_id


	def get_vocab_size(self):
		return len(self.tokens)


	def update(self, token_iterable):
		return [self.add(token) for token in token_iterable]


	def get_id(self, token):
		token = ensure_str(token)
		try:
			return self.map[token]

		# If that token isn't in the vocabulary, what should we do?
		# This depends on the setting for on_unk.  We can return the UNK
		# token silently, return the UNK token with a warning, or 
		# raise an error.
		except KeyError:
			if self.on_unk == SILENT:
				return self.map['UNK']  # i.e. return 0
			elif self.on_unk == WARN:
				print 'Warning, unrecognized token: %s' % token
				return self.map['UNK']  # i.e. return 0
			elif self.on_unk == ERROR:
				raise
			else:
				raise ValueError(
					'Unrecognized value for on_unk in TokenMap.'
				)


	def get_ids(self, token_iterable):
		return [self.get_id(token) for token in token_iterable]


	def get_token(self, idx):
		return self.tokens[idx]

	def get_tokens(self, idx_iterable):
		return [self.tokens[idx] for idx in idx_iterable]


	def __len__(self):
		return len(self.tokens)


	def save(self, filename):
		if filename.endswith('.gz'):
			f = gzip.open(filename, 'w')
		else:
			f = open(filename, 'w')

		for idx, token in enumerate(self.tokens):
			f.write(token + '\n')


	def load(self, filename):
		self.map = {}
		self.tokens = []

		if filename.endswith('.gz'):
			f = gzip.open(filename)
		else:
			f = gzip.open(filename)

		for idx, line in enumerate(f):
			token = line.strip()
			self.map[token] = idx
			self.tokens.append(token)

