# Image processing methods. This module contains a method which normalized and stretches the contrast
# and a toy implementation of dilation that is only kept to keep the file structure nice

import numpy as np 
import matplotlib.pyplot as plt
import scipy.ndimage as nd

#shows all images one after another
def showImages(imgs):
	for i in xrange(0, len(imgs)):
		plt.figure(i+1)
		plt.imshow(imgs[i])
		plt.show()

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

# This dilates the image
def dilate(img):
	img = nd.binary_dilation(img)
	return img

# A single method to call that does the entire preprocessing as determined by trial and error
# This preprocess the training images, data[0], and the test images, data[2]
def preProcess(data, threshold):
	print "\nPreprocessing..."
	print "   training data"
	for i in xrange(0, len(data[0])):
		data[0][i] = normalizeAndStretch(data[0][i], threshold)
		data[0][i] = dilate(data[0][i])
	print "   test data"
	for j in xrange(0, len(data[2])):
		data[2][j] = normalizeAndStretch(data[2][j], threshold)
		data[2][j] = dilate(data[2][j])
	print "Done."
	return data
