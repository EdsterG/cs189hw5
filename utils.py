import numpy as np
from numpy import linalg as la
import ipdb

def max_info_feature(data,y,feature_axis=1): # Assumes binary features
    '''Determine the feature X_j to split which maximizes info gain of dataset
       Info_gain_{X_j} = H(D) - \sum_{X_j=x_j) P(X_j = x_j)*H(D|X_j=x_j)'''
    if feature_axis == 1:
        data = data.T
    for feature in data:
        ipdb.set_trace()


def entropy(y):
    '''Calculate the entropy in a dataset with class output y'''
    H = 0
    y_vals = np.unique(y)
    y_len = len(y)
    for y_val in y_vals:
        p_y_val = float(sum(y == y_val))/y_len
        H -= p_y_val*np.log2(p_y_val)
    return H