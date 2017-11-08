import numpy as np
import pandas as pd
import scipy.misc # to visualize only
from sklearn import linear_model

#logistical model
logistic = linear_model.LogisticRegression(tol=0.1, max_iter=100, verbose=1)
print("loading training data x...")
trainX = np.loadtxt("train_x.csv", delimiter=",") # load from text
print("loading training data y...")
trainY = np.loadtxt("train_y.csv", delimiter=",")
print("loading test data x...")
testX = np.loadtxt("test_x.csv", delimiter=",")
#scipy.misc.imshow(x[0]) # to visualize only
print("reshaping...")
trainX = trainX.reshape(-1, 4096) # reshape
trainY = trainY.reshape(-1,1)
testX = testX.reshape(-1, 4096) # reshape

print("Fitting data...")
logistic.fit(trainX, trainY.ravel())

print("Predicting testset...")
pred = logistic.predict(testX)
pred_df = pd.DataFrame(pred)

pred_df.to_csv("predictions_test.csv", index_label="id")
