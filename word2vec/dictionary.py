'''
The dictionary provides a mapping from tokens (plain words) to integer ids,
as well as a map from ids back to words.
'''

import gzip


# TODO: enable loading and saving
class Dictionary(object):

	SILENT = 0
	WARN = 1
	ERROR = 2
	UNK = 0

	def __init__(self, on_unk=WARN):
		# Initialize.  We reserve the first token for as the token 'UNK' 
		# (for "unknown").
		self.map = {'UNK':self.UNK}	# keys are tokens, values are ids
		self.tokens = ['UNK']		# entries are tokens, indices are ids
		self.on_unk = on_unk		# determines what to do for unknown
									# 	tokens.


	def add(self, token):
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
		try:
			return self.map[token]

		# If that token isn't in the vocabulary, what should we do?
		# This depends on the setting for on_unk.  We can return the UNK
		# token silently, return the UNK token with a warning, or 
		# raise an error.
		except KeyError:
			if self.on_unk == self.SILENT:
				return self.map['UNK']  # i.e. return 0
			elif self.on_unk == self.WARN:
				print 'Warning, unrecognized token: %s' % token
				return self.map['UNK']  # i.e. return 0
			elif self.on_unk == self.ERROR:
				raise
			else:
				raise ValueError(
					'Unrecognized value for on_unk in Dictionary.'
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



