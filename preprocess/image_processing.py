# Image processing methods. This module contains a method which normalized and stretches the contrast
# and a toy implementation of dilation that is only kept to keep the file structure nice

import numpy as np 
import scipy.ndimage as nd

# This normalizes the pixel values of *ONE* image to a range of [0,1] using the formula
#
#           new_pixel = (old_pixel - min_val) / (max_val - min_val).
#
# Then sends all normalized values that are less than the threshold to 0
# and all that are above to 1. A good threshold is about 0.97
def normalizeAndStretch(img, threshold):
	min_val = np.amin(img)
	max_val = np.amax(img)
	for i in xrange(0, 64):
		for j in xrange(0, 64):
			if ((img[i][j] - min_val) / (max_val - min_val)) <= threshold:
				img[i][j] = 0.0
			else:
				img[i][j] = 1.0
	return img

# A single method to call that does the entire preprocessing.
# First normalize and set all values to binary. Then Dilate.
def preProcess(data, threshold):
	for i in xrange(0, len(data)):
		data[i] = normalizeAndStretch(data[i], threshold)
		data[i] = nd.binary_dilation(data[i])
	return data
