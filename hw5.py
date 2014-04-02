import numpy as np
import pylab as plt
import scipy.io as sio
from scipy import linalg as la

data = sio.loadmat(open("spam.mat"))

Xtrain = data['Xtrain']
Ytrain = data['ytrain']
Xtest = data['Xtest']

print Xtrain.shape
print Ytrain.shape


#class DecisionTree(Object):
#	left = None
#	right = None