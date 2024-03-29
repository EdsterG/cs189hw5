import numpy as np
from numpy import linalg as la
import ipdb
import sklearn.cross_validation as cv
import random

def kaggleSubmission(result,suffix=None):
    idRange = np.arange(1,result.shape[0]+1).reshape(result.shape)
    temp = np.concatenate((idRange,result),axis=1)
    temp = temp.astype(int)
    csvFile = np.concatenate(([['Id','Category']],temp))
    if suffix:
        fileName = 'testResults'+suffix+'.csv'
    else:
        fileName = 'testResults.csv'
    np.savetxt(fileName, csvFile, delimiter=",",fmt="%s")

# Borrowed and modified sample code from sklearn cv example page
def crossValidate(X,y,Classifier,num_folds=10,**kwargs):
    kf = cv.KFold(X.shape[0], n_folds=num_folds, shuffle=True)
    totalError = 0.0
    print "Cross validating..."
    i=1
    for train_index, test_index in kf:
        #print "TRAIN:", train_index, "TEST:", test_index
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier = Classifier(X_train,y_train,**kwargs)
        pred = classifier.classify(X_test)
        error = (pred!=y_test).sum()/float(y_test.size)
        print "Iteration %d: %0.3f" % (i,error)
        totalError += error
        i+=1
    avgError = totalError/num_folds
    print "Total Error: %0.3f" % avgError
    return avgError

H = lambda y : entropy(y)
def entropy(y):
    '''Calculate the entropy in a dataset with class output y'''
    H = 0
    y_vals = np.unique(y)
    y_len = len(y)
    for y_val in y_vals:
        p_y_val = float((y == y_val).sum())/y_len
        H -= p_y_val*np.log2(p_y_val)
    return H

infoImpurity = lambda y : entropy(y)

def giniImpurity(y):
    impurity = 1
    y_vals = np.unique(y)
    y_len = len(y)
    for y_val in y_vals:
        p_y_val = float((y == y_val).sum())/y_len
        impurity -= p_y_val**2
    return impurity

def misclassificationImpurity(y):
    if y.size==0:
        return 0
    y_vals = np.unique(y)
    p_y_val = np.zeros(y_vals.size)
    y_len = len(y)
    i = 0
    for y_val in y_vals:
        p_y_val[i] = float((y == y_val).sum())/y_len
        i+=1
    return 1-max(p_y_val)


# borrowed from 188
def nSample(distribution, values, n):
    rand = [random.random() for i in range(n)]
    rand.sort()
    distribution = distribution.copy()
    samples = []
    samplePos, distPos, cdf = 0, 0, distribution[0]
    while samplePos < n:
        if rand[samplePos] < cdf:
            samplePos += 1
            samples.append(values[distPos])
        else:
            distPos += 1
            cdf += distribution[distPos]
    return samples

def max_info_feature(data,y,H_D,feature_axis=1): # Assumes binary features
    '''Determine the feature X_j to split which maximizes info gain of dataset
       Info_gain_{X_j} = H(D) - \sum_{X_j=x_j) P(X_j = x_j)*H(D|X_j=x_j)'''
    H_D_x = np.zeros(data.shape[feature_axis])
    i = 0
    if feature_axis == 1:
        data = data.T # loop through features of matrix
    for feature in data:
        for b in {0,1}:
            idx_b = np.where(feature == b)
            p_b = float(sum(feature == b))/len(y)
            H_D_x[i] += p_b*H(y[idx_b])
        i += 1
    if max(H_D - H_D_x) == 0:
        return None # None of the features give any information gain
    return np.argmax(H_D-H_D_x)
