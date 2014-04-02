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

    def __init__(self,x,y):
        #The decision tree will recursively auto train when initialized with data (x,y)
        self.train(x,y)

        self.left = None
        self.right = None
        self.feature = None
        self.leaf = False
        self.label = None

    def train(self,x,y):
        pass

    def classify(self,x):
        if not self.leaf:
            if x[self.feature] == True:
                label = self.left.classify(x)
            else:
                label = self.right.classify(x)
            return label
        else:
            return self.label
