import theano.tensor as T
from theano import function
import lasagne


def noise_contrast(signal, noise, scale=True):
	'''
	Takes the theano symbolic variables `signal` and `noise`, whose
	elements are interpreted as probabilities, and creates a loss
	function which rewards large probabilities in signal and penalizes 
	large probabilities in noise.
	
	`signal` and `noise` can have any shape.  Their contributions will be 
	summed over all dimensions.

	If `scale` is true, then scale the loss function by the size of the
	signal tensor --- i.e. divide by the signal batch size.  This makes
	the scale of the loss function invariant to changes in batch size
	'''

	signal_score = T.log(signal).sum()
	noise_score = T.log(1-noise).sum()
	objective = signal_score + noise_score
	loss = -objective

	loss = loss / signal.shape[0]
	
	return loss


class NoiseContraster(object):

	def __init__(
		self,
		signal_input,
		noise_input,
		learning_rate=0.1,
		momentum=0.9
	):
		self.signal_input = signal_input
		self.noise_input = noise_input
		self.learning_rate = learning_rate
		self.momentum=momentum

		self.combined = T.concatenate([
			self.signal_input, self.noise_input
		])

	def get_combined_input(self):
		return self.combined

	def get_train_func(self, output, params):

		# Split the output based on the size of the individual input streams
		self.signal_output = output[:self.signal_input.shape[0]]
		self.noise_output = output[self.signal_input.shape[0]:]

		# Construct the loss based on Noise Contrastive Estimation
		self.loss = noise_contrast(self.signal_output, self.noise_output)

		# Get the parameter updates using stochastic gradient descent with
		# nesterov momentum.  TODO: make the updates configurable
		self.updates = lasagne.updates.nesterov_momentum(
			self.loss,
			params,
			self.learning_rate,
			self.momentum
		)

		self.train = function(
			inputs=[self.signal_input, self.noise_input], 
			outputs=self.loss,
			updates=self.updates
		)

		return self.train



