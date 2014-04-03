from utils import *

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
