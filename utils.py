from __future__ import division
import numpy as np
from numpy import linalg as la
import ipdb

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
        #ipdb.set_trace()
        return None # None of the features give any information gain
    return np.argmax(H_D-H_D_x)

H = lambda y : entropy(y)
def entropy(y):
    '''Calculate the entropy in a dataset with class output y'''
    H = 0
    y_vals = np.unique(y)
    y_len = len(y)
    for y_val in y_vals:
        p_y_val = float(sum(y == y_val))/y_len
        H -= p_y_val*np.log2(p_y_val)
    return H

class DecisionTree:
    def __init__(self,data,y):
        #The decision tree will recursively auto train when initialized with data (x,y)
        self.left = None
        self.right = None
        self.featureIndex = None
        self.leaf = False
        self.label = None

        self.train(data,y)

    def train(self,data,y):
        H_y = H(y)
        featureIndex = max_info_feature(data,y,H_y)
        sum_y = y.sum()
        #If all labels the same, create leaf
        if featureIndex == None:
            self.leaf = True
            self.label = np.round(sum_y/y.shape[0])
        #If entropy gain is zero, create leaf
        elif sum_y == y.shape[0] or sum_y == 0:
            self.leaf = True
            self.label = y[0]
        else:
            idx_0 = np.where(data[:,featureIndex] == 0)
            X_0 = data[idx_0]
            y_0 = y[idx_0]
            idx_1 = np.where(data[:,featureIndex] == 1)
            X_1 = data[idx_1]
            y_1 = y[idx_1]
            self.featureIndex = featureIndex
            self.left = DecisionTree(X_1,y_1)
            self.right = DecisionTree(X_0,y_0)

    def classify(self,data):
        y = np.zeros((data.shape[0],1))
        i = 0
        for point in data:
            y[i] = self.getPointLabel(point)
            i += 1
        return y

    def getPointLabel(self,point):
        if not self.leaf:
            if point[self.featureIndex] == True:
                label = self.left.getPointLabel(point)
            else:
                label = self.right.getPointLabel(point)
            return label
        else:
            return self.label
