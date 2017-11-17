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
import csv

#####################
# FF NEURAL NETWORK #
#####################

def calculate_loss(model, num_hl):

	in_hp_matrix = model['in_hp_matrix']
	in_hp_vector = model['in_hp_vector']
	out_hp_matrix = model['out_hp_matrix']
	out_hp_vector = model['out_hp_vector']
	hl_hp_matrix = [0 for i in range(num_hl)]
	hl_hp_vector = [0 for i in range(num_hl)]
	for i in range (num_hl):
		hl_hp_matrix[i] = model['hl_hp_matrix_' + str(i)]
		hl_hp_vector[i] = model['hl_hp_vector_' + str(i)]

	# Forward propagation
	in_matrix_ops = x.dot(in_hp_matrix) + in_hp_vector
	in_activation = np.tanh(in_matrix_ops)

	hl_matrix_ops = [0 for i in range(num_hl)]
	hl_activation = [0 for i in range(num_hl)]
	for i in range(num_hl):
		if (i < 1): # if it's the first hidden layer
			hl_matrix_ops[i] = in_activation.dot(hl_hp_matrix[i]) + hl_hp_vector[i]
			hl_activation[i] = np.tanh(hl_matrix_ops[i])
		else:
			hl_matrix_ops[i] = hl_activation[i-1].dot(hl_hp_matrix[i]) + hl_hp_vector[i]
			hl_activation[i] = np.tanh(hl_matrix_ops[i])

	out_matrix_ops = hl_activation[num_hl-1].dot(out_hp_matrix) + out_hp_vector
	exp_scores = np.exp(out_matrix_ops)
	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

	# Calculating the loss
	correct_logprobs = -np.log(probs[range(num_examples), y])
	data_loss = np.sum(correct_logprobs)
	# Add regularization term to loss
	regularization_strength = 0.01
	data_loss += regularization_strength/2 * (np.sum(np.square(in_hp_matrix)) + np.sum(np.square(out_hp_matrix)))

	return 1./num_examples * data_loss

def predict(model, inp_matrix, num_hl):

	in_hp_matrix = model['in_hp_matrix']
	in_hp_vector = model['in_hp_vector']
	out_hp_matrix = model['out_hp_matrix']
	out_hp_vector = model['out_hp_vector']
	hl_hp_matrix = [0 for i in range(num_hl)]
	hl_hp_vector = [0 for i in range(num_hl)]
	for i in range (num_hl):
		hl_hp_matrix[i] = model['hl_hp_matrix_' + str(i)]
		hl_hp_vector[i] = model['hl_hp_vector_' + str(i)]

	# Forward propagation
	in_matrix_ops = inp_matrix.dot(in_hp_matrix) + in_hp_vector
	in_activation = np.tanh(in_matrix_ops)

	hl_matrix_ops = [0 for i in range(num_hl)]
	hl_activation = [0 for i in range(num_hl)]
	for i in range(num_hl):
		if (i < 1): # if it's the first hidden layer
			hl_matrix_ops[i] = in_activation.dot(hl_hp_matrix[i]) + hl_hp_vector[i]
			hl_activation[i] = np.tanh(hl_matrix_ops[i])
		else:
			hl_matrix_ops[i] = hl_activation[i-1].dot(hl_hp_matrix[i]) + hl_hp_vector[i]
			hl_activation[i] = np.tanh(hl_matrix_ops[i])

	out_matrix_ops = hl_activation[num_hl-1].dot(out_hp_matrix) + out_hp_vector
	exp_scores = np.exp(out_matrix_ops)
	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

	return np.argmax(probs, axis=1)

# Returns the NN model after learning parameters
def build_model(num_nodes, num_hl):

	num_passes = 10000 # to save some time, keep it low
	regularization_strength = 0.01
	model = {}

	# Set the initial hyperparameters
	np.random.seed(3) # This gave the lowest loss with 3 HLs
	in_hp_matrix = np.random.randn(in_dim, num_nodes) / np.sqrt(in_dim)
	in_hp_vector = np.zeros((1, num_nodes))

	# Hidden layers
	hl_hp_matrix = [0 for i in range(num_hl)]
	hl_hp_vector = [0 for i in range(num_hl)]

	for i in range(num_hl):
		hl_hp_matrix[i] = np.random.randn(num_nodes, num_nodes) / np.sqrt(num_nodes)
		hl_hp_vector[i] = np.zeros((1, num_nodes))

	out_hp_matrix = np.random.randn(num_nodes, out_dim) / np.sqrt(num_nodes)
	out_hp_vector = np.zeros((1, out_dim))

	learning_rate = 0.01

	# Gradient descent
	for gd in range(0, num_passes):

		# Forward propagation
		in_matrix_ops = x.dot(in_hp_matrix) + in_hp_vector
		in_activation = np.tanh(in_matrix_ops)

		hl_matrix_ops = [0 for i in range(num_hl)]
		hl_activation = [0 for i in range(num_hl)]
		for i in range(num_hl):
			if (i < 1): # if it's the first hidden layer
				hl_matrix_ops[i] = in_activation.dot(hl_hp_matrix[i]) + hl_hp_vector[i]
				hl_activation[i] = np.tanh(hl_matrix_ops[i])
			else:
				hl_matrix_ops[i] = hl_activation[i-1].dot(hl_hp_matrix[i]) + hl_hp_vector[i]
				hl_activation[i] = np.tanh(hl_matrix_ops[i])

		out_matrix_ops = hl_activation[num_hl-1].dot(out_hp_matrix) + out_hp_vector
		exp_scores = np.exp(out_matrix_ops)
		probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

		# Backpropagation
		delta_out = probs
		delta_out[range(num_examples), y] -= 1
		d_out_hp_matrix = (hl_activation[num_hl - 1].T).dot(delta_out)
		d_out_hp_vector = np.sum(delta_out, axis=0, keepdims=True)

		delta_hl = [0 for i in range(num_hl)]
		d_hl_hp_matrix = [0 for i in range(num_hl)]
		d_hl_hp_vector = [0 for i in range(num_hl)]
		for i in reversed(range(num_hl)):
			if (i == num_hl - 1): # if top level hidden layer
				delta_hl[i] = delta_out.dot(out_hp_matrix.T) * (1 - np.power(hl_activation[i], 2))
				d_hl_hp_matrix[i] = (hl_activation[i-1].T).dot(delta_hl[i])
				d_hl_hp_vector[i] = np.sum(delta_hl[i], axis=0, keepdims=True)
			elif(i == 0): # if bottom level hidden layers
				delta_hl[i] = delta_hl[i+1].dot(hl_hp_matrix[i+1].T) * (1 - np.power(hl_activation[i], 2))
				d_hl_hp_matrix[i] = (in_activation.T).dot(delta_hl[i])
				d_hl_hp_vector[i] = np.sum(delta_hl[i], axis=0, keepdims=True)
			else:
				delta_hl[i] = delta_hl[i+1].dot(hl_hp_matrix[i+1].T) * (1 - np.power(hl_activation[i], 2))
				d_hl_hp_matrix[i] = (hl_activation[i-1].T).dot(delta_hl[i])
				d_hl_hp_vector[i] = np.sum(delta_hl[i], axis=0, keepdims=True)

		delta_in = delta_hl[0].dot(hl_hp_matrix[0].T) * (1 - np.power(in_activation, 2))
		d_in_hp_matrix = np.dot(x.T, delta_in)
		d_in_hp_vector = np.sum(delta_in, axis=0)

		# Add regularization terms
		d_out_hp_matrix += regularization_strength * out_hp_matrix
		d_hl_hp_matrix = [0 for i in range(num_hl)]
		for i in range (num_hl):
			d_hl_hp_matrix[i] = regularization_strength * hl_hp_matrix[i]
		d_in_hp_matrix += regularization_strength * in_hp_matrix

		# Update the parameters
		in_hp_matrix += -learning_rate * d_in_hp_matrix
		in_hp_vector += -learning_rate * d_in_hp_vector
		for i in range (num_hl):
			hl_hp_matrix[i] += -learning_rate * d_hl_hp_matrix[i]
			hl_hp_vector[i] += -learning_rate * d_hl_hp_vector[i]
		out_hp_matrix += -learning_rate * d_out_hp_matrix
		out_hp_vector += -learning_rate * d_out_hp_vector

		# Assign new parameters to the model
		model = { 'in_hp_matrix': in_hp_matrix,
		'in_hp_vector': in_hp_vector,
		'out_hp_matrix': out_hp_matrix,
		'out_hp_vector': out_hp_vector }
		for i in range (num_hl):
			model['hl_hp_matrix_' + str(i)] = hl_hp_matrix[i]
			model['hl_hp_vector_' + str(i)] = hl_hp_vector[i]

		# Print loss which should be getting lower after every iteration
		if gd % 1000 == 0:
			print("Iteration " + str(gd) + " loss: " + str(calculate_loss(model, num_hl)))

	return model

####################
###     MAIN     ###
####################

num_examples = 10000 # reduced from 50000 due to memory errors
size_testset = 10000
num_rows = 64
num_columns = 64

# Load training and test data

print('Loading Data...')

x_in = open('data/x_train_data.pkl', 'rb')
y_in = open('data/y_train_data.pkl', 'rb')
x_test = open('data/x_test_data.pkl', 'rb')

x = pickle.load(x_in)
print('Loaded x data.')

y = pickle.load(y_in)
print('Loaded y data.')

x_test = pickle.load(x_test)
print('Loaded test data.')

# Prepare x and y for the NN

print('Preparing x and y for NN...')

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

in_dim = 64 * 64 # how many pixels there are
out_dim = len(outcomes_vector)

# Build the NN

print('Building the NN model...')
num_hl = 3
model = build_model(len(outcomes_vector), num_hl)

# Predict using the NN

print('Preparing test set...')
test_examples = []
for example in range (0, num_examples):
	new_example = []
	for row in range (0, num_rows):
		for column in range (0, num_columns):
			new_example.append(x_test[example][row][column])
	test_examples.append(new_example)

x_test = np.asarray(test_examples, dtype=np.float32)

print('Predicting for test set...')
predictions = []
for i in range (0, size_testset):
	predictions.append(outcomes_vector[predict(model, x_test[i], num_hl)[0]])

print('Writing predictions...')
with open('prediction/nn_predictions.csv', 'w') as predict_file:
	fieldnames = ['Id', 'Category']
	writer = csv.DictWriter(predict_file, fieldnames=fieldnames)
	writer.writeheader()
	for i in range(len(predictions)):
		writer.writerow({'Id': i + 1, 'Category': predictions[i]})

print "Done!"
