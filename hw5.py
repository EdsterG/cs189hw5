import numpy as np
import pylab as plt
import scipy.io as sio
from scipy import linalg as la
from utils import *
from RandomForest import RandomForest
from DecisionTree import DecisionTree
from AdaBoost import AdaBoost

# import dataset
data = sio.loadmat(open("spam.mat"))
X = data['Xtrain']
y = data['ytrain']
Xtest = data['Xtest']

# binarize features for easy decision tree use
X = (X > 0) # convert train/test into matrix of logicals
Xtest = (Xtest > 0)
y = (y > 0)

def decision_tree():
    print "Initilizaing/Training decision tree"
    dt = DecisionTree(X,y)
    print "Traning Complete"

    print "Classifying training set"
    pred = dt.classify(X)

    print "Training set error: "+str((pred!=y).sum()/float(y.size))

def random_forest(M):
    print "Initializing/Training Random Forest"
    rf = RandomForest(X,y,M)
    print "Traning Complete"

    print "Classifying training set"
    pred = rf.classify(X)

    print "Training set error: "+str((pred!=y).sum()/float(y.size))

def adaboost():
    print "Initializing/Training AdaBoost"
    ab = AdaBoost(X,y)
    print "trained"

    print "Classifying training set"
    pred = ab.classify(X)

    print "Training set error: "+str((pred!=y).sum()/float(y.size))

def cross_validation():
	crossValidate(X,y,DecisionTree)
	#sliceLocation = 1000
	#Xtrain = X[:sliceLocation]
	#Ytrain = y[:sliceLocation]
	#Xtest = X[sliceLocation:]
	#Ytest = y[sliceLocation:]
	#dt = DecisionTree(Xtrain,Ytrain)
	#pred = dt.classify(Xtest)
	#print "Training set error: "+str((pred!=Ytest).sum()/float(y.size))

if __name__ == '__main__':
    #decision_tree()
    #crossValidate(X,y,DecisionTree)
    #random_forest(M=4)
    #adaboost()
    crossValidate(X,y,AdaBoost)
