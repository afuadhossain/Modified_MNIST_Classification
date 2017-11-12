#A script which reads in the csv files and saves them 
# as faster-to-access pkl files.

import numpy as np
import pickle
import image_processing as ip

#import the data, preprocess, and save as a numpy pickle array for speed

print "\nPreprocessing data..."

#for x_train
print "\ntrain_x.csv"
print "   reading"
x = np.loadtxt("data/train_x.csv", delimiter=",") # load from text
x = x.reshape(-1,64,64)
print "   preprocessing"
x = ip.preProcess(x, 0.7)
print "   dumping to data/pkl.x_train_data.pkl"
x_out = open('data/x_train_data.pkl','wb')
pickle.dump(x, x_out)
x_out.close()

#for y_train
print "\ntrain_y.csv"
print "   reading"
y = np.loadtxt("data/train_y.csv", delimiter=",")
y = y.reshape(-1)
print "   dumping to data/y_train_data.pkl"
y_out = open("data/y_train_data.pkl",'wb')
pickle.dump(y, y_out)
y_out.close()

#for x_test
print "\ntest_x.csv"
print "   reading"
t = np.loadtxt("data/test_x.csv", delimiter=",")
t = t.reshape(-1,64,64)
print "   preprocessing"
t = ip.preProcess(t, 0.7)
print "   dumping to data/x_test_data.pkl"
t_out = open("data/x_test_data.pkl",'wb')
pickle.dump(t, t_out)
t_out.close()

print "\nDone\n"