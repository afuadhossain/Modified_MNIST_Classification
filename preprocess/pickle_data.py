import numpy as np
import pickle

#import the data and save as a numpy pickle array for fastness
#for x_train

print "\nreading train_x..."

x = np.loadtxt("../data/train_x.csv", delimiter=",") # load from text
x = x.reshape(-1,64,64)
x_out = open('x_train_data.pkl', 'wb')
pickle.dump(x, x_out)
x_out.close()

print "reading train_y..."

#for y_train
y = np.loadtxt("../data/train_y.csv", delimeter=",")
y = y.reshape(-1,1)
y_out = open("y_train_data.pkl", delimeter=",")
pickle.dump(y, y_out)
y_out.close()

print "reading test_x...\n"

#for x_test
t = np.loadtxt("../data/test_x.csv", delimeter=",")
t = t.reshape(-1,64,64)
t_out = open("x_test_data.pkl", delimeter=",")
pickle.dump(t, t_out)
t_out.close()