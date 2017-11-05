#Basic File Reading for image data.

import numpy as np 
#for reading in all training and test data. returns a list of the form [train_x, train_y, test].
def readFiles(train_x, train_y, test):
	print "\nReading..."
	print "   training data"
	x = np.loadtxt(train_x, delimiter=",") # load from text
	x = x.reshape(-1, 64, 64) # reshape 
	y = np.loadtxt(train_y, delimiter=",") 
	y = y.reshape(-1, 1)
	print "   test data"
	temp = np.loadtxt(test, delimiter=",")
	t = []
	for i in xrange(0, len(temp)):
		t.append( temp[i].reshape(64,64) )
	data = [x,y,t]
	print "Done."
	return data