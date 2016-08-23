import os

# Only import theano and lasagne if environment permits it
exclude_theano_set = 'EXCLUDE_THEANO' in os.environ
if exclude_theano_set and int(os.environ['EXCLUDE_THEANO']) == 1:
	# Don't import theano and lasagne
	pass
else:
	# Do import theano and lasagne
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


def get_noise_contrastive_loss(activation, num_signal, scale=True):
	'''
	Convenience function to get the noise_contrast expression by specifying a
	single batch of outputs and giving the number of entries along axis 0,
	corresponding to signal activations (all others are assumed to be noise)
	activations.

	Differs from noise_contrast only in that noise contrast expects two
	theano variables, one for signal activations and one for noise.  This means
	the caller would have to separate signal from noise first.
	'''
	signal_activation = activation[0:num_signal,]
	noise_activation = activation[num_signal:,]
	return noise_contrast(signal_activation, noise_activation, scale)
