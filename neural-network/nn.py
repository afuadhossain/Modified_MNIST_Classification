# Feedforward Neural Network

# Guidelines
# A fully connected feedforward neural network (from lecture 14), trained by
# backpropagation, where the network architecture (number of nodes / layers),
# learning rate and termination are determined by cross-validation. This method
# must be fully implemented by your team, and corresponding code submitted.

# Note to TA
# Please note that the below code was inspired from the 3-layer neural network
# Python guide found here:
# http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
# as well as the COMP 551 lecture 14 notes found here:
# http://cs.mcgill.ca/~jpineau/comp551/Lectures/14NeuralNets.pdf

import numpy as np
import pickle

########
# MAIN #
########

num_examples = 10000
num_rows = 64
num_columns = 64

# Load training and eval data

print('Loading Data...')

x_in = open('data/x_train_data.pkl', 'rb')
y_in = open('data/y_train_data.pkl', 'rb')

x = pickle.load(x_in)
print('Loaded x data.')

y = pickle.load(y_in)
print('Loaded y data.')

# Prepare x and y for the NN

all_examples = []
for example in range (0, num_examples):
	new_example = []
	for row in range (0, num_rows):
		for column in range (0, num_columns):
			new_example.append(x[example][row][column])
	all_examples.append(new_example)

x = np.asarray(all_examples, dtype=np.float32)

outcomes_vector = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]

for example in y:
	mapped_y = outcomes_vector.index(example)

y = np.asarray(mapped_y, dtype=np.int32)

input_dim = 64 * 64 # how many pixels there are
output_dim = len(outcomes_vector)

####################
# HELPER FUNCTIONS #
####################

def calculate_loss(model):
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
	# Forward propagation
	z1 = x.dot(W1) + b1
	a1 = np.tanh(z1)
	z2 = a1.dot(W2) + b2
	exp_scores = np.exp(z2)
	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
	# Calculating the loss
	correct_logprobs = -np.log(probs[range(num_examples), y])
	data_loss = np.sum(correct_logprobs)
	return 1./num_examples * data_loss

def predict(model, x):
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
	# Forward propagation
	z1 = x.dot(W1) + b1
	a1 = np.tanh(z1)
	z2 = a1.dot(W2) + b2
	exp_scores = np.exp(z2)
	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
	return np.argmax(probs, axis=1)

# Returns the NN model after learning parameters
def build_model(num_nodes):

	num_passes = 1000 # to save some time, keep it low
	regularization_strength = 0.01
	model = {}

	# Initialize the parameters, which we will eventually learn
	np.random.seed(0)
	W1 = np.random.randn(input_dim, num_nodes) / np.sqrt(input_dim)
	b1 = np.zeros((1, num_nodes))
	W2 = np.random.randn(num_nodes, output_dim) / np.sqrt(num_nodes)
	b2 = np.zeros((1, output_dim))

	learning_rate = 0.01

	# Gradient descent
	for i in range(0, num_passes):

		# Forward propagation
		z1 = x.dot(W1) + b1
		a1 = np.tanh(z1)
		z2 = a1.dot(W2) + b2
		exp_scores = np.exp(z2)
		probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

		# Backpropagation
		delta3 = probs
		delta3[range(num_examples), y] -= 1
		dW2 = (a1.T).dot(delta3)
		db2 = np.sum(delta3, axis=0, keepdims=True)
		delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
		dW1 = np.dot(x.T, delta2)
		db1 = np.sum(delta2, axis=0)

		# Add regularization terms
		dW2 += regularization_strength * W2
		dW1 += regularization_strength * W1

		# Gradient descent parameter update
		W1 += -learning_rate * dW1
		b1 += -learning_rate * db1
		W2 += -learning_rate * dW2
		b2 += -learning_rate * db2

		# Assign new parameters to the model
		model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

		# Print loss which should be getting lower after every iteration
		if i % 50 == 0:
			print("Loss after iteration %i: %f" %(i, calculate_loss(model)))

	return model

# MAIN CONTINUED

model = build_model(len(outcomes_vector))

print "Done!"
