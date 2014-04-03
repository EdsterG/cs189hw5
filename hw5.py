from __future__ import division
import numpy as np
import pylab as plt
import scipy.io as sio
from scipy import linalg as la
from utils import *
from RandomForest import RandomForest

# import dataset
data = sio.loadmat(open("spam.mat"))

X = data['Xtrain']
y = data['ytrain']
Xtest = data['Xtest']


# binarize features for easy decision tree use
X = (X > 0) # convert train/test into matrix of logicals
Xtest = (Xtest > 0)
y = (y > 0)

#calculate H(X)
#data_entropy = H(y) # ~0.97
#print data_entropy

def decision_tree():
    print "Initilizaing/Training decision tree"
    dt = DecisionTree(X,y)
    print "Traning Complete"

    print "Classifying training set"
    pred = dt.classify(X)

    print "Training set error: "+str((pred!=y).sum()/y.size)

def random_forest(M):
    print "Initializing/Training Random Forest"
    rf = RandomForest(X,y,M)
    print "Traning Complete"

    print "Classifying training set"
    pred = rf.classify(X)

    print "Training set error: "+str((pred!=y).sum()/y.size)

if __name__ == '__main__':
    # decision_tree()
    random_forest(M=4)