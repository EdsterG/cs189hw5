from __future__ import division
import numpy as np
import pylab as plt
import scipy.io as sio
from scipy import linalg as la
from utils import *

# import dataset
data = sio.loadmat(open("spam.mat"))

X = data['Xtrain']
y = data['ytrain']
Xtest = data['Xtest']
<<<<<<< HEAD
ipdb.set_trace()
# print train.shape
# print target.shape
# print test.shape
=======
>>>>>>> cd8f55446de7aa984949c836f869f6bf1994df8e

# binarize features for easy decision tree use
X = (X > 0) # convert train/test into matrix of logicals
Xtest = (Xtest > 0)
y = (y > 0)

#calculate H(X)
#data_entropy = H(y) # ~0.97
#print data_entropy


print "Initilizaing/Training decision tree"
dt = DecisionTree(X,y)
print "Finished training"

print "Classifying training set"
pred = dt.classify(X)

print "Training set error:"+str((pred!=y).sum()/y.size)
