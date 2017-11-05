# A sample main method for reading and preprocessing data
# This reads in the training and test files, and preprocesses the images of both training and test data
# Takes upwards of 10mins to run... lots of data here.

import image_processing as ip
import read_files as rf

def main():
	# training and test files 
	train_x = "../data/train_x.csv"
	train_y = "../data/train_y.csv"
	test = "../data/test_x.csv"

	# read the training and test files into a list of form [train_x, train_y, test]
	# then preprocess the data
	data = rf.readFiles(train_x, train_y, test)
	data = ip.preProcess(data, 0.97)

main()
