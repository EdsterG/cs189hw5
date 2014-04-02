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

# print train.shape
# print target.shape
# print test.shape

# binarize features for easy decision tree use
X = (X > 0) # convert train/test into matrix of logicals
Xtest = (Xtest > 0)

# calculate H(X)
data_entropy = entropy(y) # ~0.97






#class DecisionTree(Object):
#	left = None
#	right = None