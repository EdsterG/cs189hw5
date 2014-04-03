from utils import *

class RandomForest:

    def __init__(self,X,y,M): # M is number of trees
        self.trees = []
        self.X = X
        self.y = y
        self.randPoints = np.array([])
        self.randFeatures = np.array([])

    def train(self):
        self.randPoints = np.random.randint(X.shape[0],size=np.random.randint(1,X.shape[0]))
        self.randFeatures = np.random.randint(X.shape[1],size=np.random.randint(1,X.shape[1]))
        self.trees = [DecisionTree(X[randPoints[i],:][:,randFeatures[i]],y[randPoints[i]]) for i in range(M)]

    def classify(self,Xtest):
        predictions = np.zeros(Xtest.shape[0])
        i = 0
        for tree in self.trees:
            predictions[i] = tree.classify(Xtest[randPoints[i],:][:,randFeatures[i]])
        prediction = np.round(np.mean(predictions))
        return predicition