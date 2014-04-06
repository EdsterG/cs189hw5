from __future__ import print_function
from utils import *
from DecisionTree import DecisionTree

class RandomForest:

    def __init__(self,X,y,M): # M is number of trees
        self.trees = []
        self.X = X
        self.y = y
        self.M = M
        self.randPoints = np.array([])
        self.randFeatures = np.array([])

        self.train(X,y)

    def train(self,X,y):
        self.randPoints = [np.random.randint(X.shape[0],size=np.random.randint(1,X.shape[0])) for _ in range(self.M)]
        self.randFeatures = [np.random.choice(np.r_[:X.shape[1]],size=np.random.randint(1,X.shape[1]),replace=False) for _ in range(self.M)]
        for featureSet in self.randFeatures:
            if not (len(featureSet) == len(np.unique(featureSet))):
                print("Sampling features with replacement for some reason")
        self.trees = [print(i) or DecisionTree(X[self.randPoints[i],:][:,self.randFeatures[i]],y[self.randPoints[i]]) for i in range(self.M)]

    def classify(self,Xtest):
        predictions = np.zeros((Xtest.shape[0],self.M))
        i = 0
        for tree in self.trees:
            pred = tree.classify(Xtest[:,self.randFeatures[i]])
            pred.shape = (pred.shape[0],)
            predictions[:,i] = pred
            i+=1
        prediction = np.round(np.mean(predictions,axis=1))
        prediction.shape = (prediction.shape[0],1)
        return prediction