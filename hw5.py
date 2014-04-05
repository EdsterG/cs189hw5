import numpy as np
import pylab as plt
import scipy.io as sio
from scipy import linalg as la
from utils import *
from RandomForest import RandomForest
from DecisionTree import DecisionTree
from AdaBoost import AdaBoost
import pickle

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

def random_forest(M,kaggle=False):
    print "Initializing/Training Random Forest"
    rf = RandomForest(X,y,M)
    print "Traning Complete"

    print "Classifying training set"
    pred = rf.classify(X)

    print "Training set error: "+str((pred!=y).sum()/float(y.size))

    y_hat = rf.classify(Xtest)
    if kaggle:
        np.save('y_hat.npy', y_hat)
        with open('randomForest.txt', 'wb') as output:
            pickle.dump(rf, output, pickle.HIGHEST_PROTOCOL)

def adaboost():
    print "Initializing/Training AdaBoost"
    ab = AdaBoost(X,y)
    print "trained"

    print "Classifying training set"
    pred = ab.classify(X)

    print "Training set error: "+str((pred!=y).sum()/float(y.size))

def kaggleSubmission(result):
    # classifier = Classifier(X,y)
    # result = classifier.classify(Xtest)
    csvFile = np.concatenate(([['Id','Category']],result))
    np.savetxt("testResults.csv", csvFile, delimiter=",",fmt="%s")

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
    random_forest(M=1000,kaggle=True)
    #adaboost()
    # crossValidate(X,y,AdaBoost)
