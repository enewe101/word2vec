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
		parse=default_parse
	):
		self.files = files
		self.directories = directories
		self.skip = skip
		self.parse = default_parse


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

				for line in default_parse(filename):
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

					for line in default_parse(filename):
						queue.put(line)

		# Notify the parent process that you're done
		if pipe is not None:
			pipe.send(self.DONE)




