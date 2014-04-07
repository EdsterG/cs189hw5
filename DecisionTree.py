from utils import *

class DecisionTree:

    def __init__(self,data,labels,features=None,depth=None):
        #The decision tree will recursively auto train when initialized with data (x,y)
        self.left = None
        self.right = None
        self.featureIndex = None
        self.leaf = False
        self.label = None

        self.train(data,labels)

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


class DecisionTree2:
    ALL = None

    def __init__(self,data,labels,validFeatures=ALL,impurityMeasure=infoImpurity,maxDepth=None,sampleFeatures=False,sampleData=False,prune=False):
        #The decision tree will recursively auto train when initialized with data (x,y)
        self.sampleFeatures = sampleFeatures
        self.sampleData = sampleData
        self.maxDepth = maxDepth
        self.impurityMeasure = impurityMeasure
        self.prune = prune

        if validFeatures == self.ALL or validFeatures.size>data.shape[1]:
            self.validFeatures = np.arange(data.shape[1])
        else:
            self.validFeatures = validFeatures

        if self.sampleData == True:
            #sampleSize = np.random.randint(1,data.shape[0])
            sampleSize = data.shape[0]
            dataIndices = np.random.randint(data.shape[0],size=sampleSize)
        else:
            dataIndices = np.arange(data.shape[0])
        data = data[dataIndices,:]
        labels = labels[dataIndices]
        self.root = Node(data,labels,None,self,1)

        #After the full tree is built it can be pruned up from the leaves.
        #Every node has two children pointers and a parent pointer to make pruning easier.

    def classify(self,data):
        y = np.zeros((data.shape[0],1))
        i = 0
        for point in data:
            y[i] = self.root.classify(point)
            i += 1
        return y

class Node:
    def __init__(self,data,labels,parent,tree,depth):
        self.data = data
        self.labels = labels
        self.tree = tree
        self.parent = parent
        self.depth = depth

        self.left = None
        self.right = None

        self.featureIndex = None

        self.label = None

        self.train(data,labels)

    def train(self,data,labels):
        if self.tree.sampleFeatures == True:
            #sampleSize = np.random.randint(1,data.shape[1])
            sampleSize = int(np.log2(data.shape[1]))+1
            featuresToUse = np.random.choice(self.tree.validFeatures,size=sampleSize,replace=False)
        else:
            featuresToUse = self.tree.validFeatures

        # If max depth was reached, stop splitting
        if self.tree.maxDepth and self.depth == self.tree.maxDepth:
            self.label = np.round(self.labels.sum()/self.labels.shape[0])
            return

        numPosLabels = labels.sum()
        # If all labels the same stop splitting
        if numPosLabels == labels.shape[0] or numPosLabels == 0:
            self.label = np.round(self.labels.sum()/self.labels.shape[0])
            return

        featureIndex = self.minImpurityFeature(data,labels,featuresToUse)
        # If entropy gain is zero stop splitting
        if featureIndex == None:
            self.label = np.round(self.labels.sum()/self.labels.shape[0])
            return

        idx_1 = np.where(data[:,featureIndex] == 1)
        X_1 = data[idx_1]
        y_1 = labels[idx_1]
        idx_0 = np.where(data[:,featureIndex] == 0)
        X_0 = data[idx_0]
        y_0 = labels[idx_0]
        self.featureIndex = featureIndex

        self.left = Node(X_1,y_1,self,self.tree,self.depth+1)
        self.right = Node(X_0,y_0,self,self.tree,self.depth+1)

    def minImpurityFeature(self,data,labels,validFeatures,feature_axis=1):
        # Assumes binary features
        '''Determine the feature X_j to split which maximizes info gain of dataset
           Info_gain_{X_j} = H(D) - \sum_{X_j=x_j) P(X_j = x_j)*H(D|X_j=x_j)'''
        currImpurity = self.tree.impurityMeasure(labels)
        impurityOfFeature = np.zeros(validFeatures.shape)
        i = 0
        if feature_axis == 1:
            data = data.T # loop through features of matrix
        for feature in data[validFeatures,:]:
            #featureIndex = validFeatures[i]
            for val in {0,1}:
                idx_val = np.where(feature == val)
                p_val = float((feature == val).sum())/labels.size
                impurityOfFeature[i] += p_val*self.tree.impurityMeasure(labels[idx_val])
            i += 1
        impurityRemoved = currImpurity-impurityOfFeature
        if max(impurityRemoved) <= 0:
            return None # None of the features give any information gain
        idx = np.argmax(impurityRemoved)
        return validFeatures[idx]

    def classify(self,point):
        if self.left and self.right:
            if point[self.featureIndex] == True:
                label = self.left.classify(point)
            else:
                label = self.right.classify(point)
            return label
        else:
            return self.label