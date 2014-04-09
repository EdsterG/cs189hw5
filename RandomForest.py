from __future__ import print_function
from utils import *
from DecisionTree import *

class RandomForest:

    def __init__(self,X,y,M=100):
        self.trees = []
        self.M = M

        self.train(X,y)

    def train(self,X,y):
        self.trees = [print(i) or DecisionTree(X,y,sampleFeatures=True,sampleData=True) for i in range(self.M)]
        #self.trees = [DecisionTree(X,y,sampleFeatures=True,sampleData=True) for i in range(self.M)]

    def classify(self,data):
        predictions = np.zeros((data.shape[0],self.M))
        i = 0
        for tree in self.trees:
            print(i)
            pred = tree.classify(data)
            pred.shape = (pred.shape[0],)
            predictions[:,i] = pred
            i+=1
        prediction = np.round(np.mean(predictions,axis=1))
        prediction.shape = (prediction.shape[0],1)
        return prediction