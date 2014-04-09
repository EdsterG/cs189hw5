from utils import *

from sklearn.cross_validation import train_test_split
# only used for experimental pruning

class DecisionTree:
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

        if self.sampleData:
            #sampleSize = np.random.randint(1,data.shape[0])
            sampleSize = data.shape[0]*0.8
            dataIndices = np.random.randint(data.shape[0],size=sampleSize)
        else:
            dataIndices = np.arange(data.shape[0])
        data = data[dataIndices,:]
        labels = labels[dataIndices]

        if not self.prune:
            self.root = Node(data,labels,None,self,1)
        else:
            # for pruning, split into train and validate
            trainSize = data.shape[0] * .7
            trainData, valData, trainLab, valLab = train_test_split(
                data, labels, test_size=0.33, random_state=42)
            # train on training partition
            self.root = Node(trainData, trainLab, None, self, 1)
            self.pruneClassify(valData, valLab)
            self.root.prune()
        
        #Every node has two children pointers and a parent pointer to make pruning easier.

    def classify(self,data):
        y = np.zeros((data.shape[0],1))
        i = 0
        for point in data:
            y[i] = self.root.classify(point)
            i += 1
        return y

    def pruneClassify(self,data, labels):
        y = np.zeros((data.shape[0],1))
        i = 0
        for point in data:
            y[i] = self.root.pruneClassify(point, labels[i])
            i += 1
        return y

class Node:
    def __init__(self,data,labels,parent,tree,depth):
        self.tree = tree
        self.parent = parent
        self.depth = depth

        self.left = None
        self.right = None

        self.featureIndex = None

        self.label = None

        # count examples that pass through each node
        self.actualCount = [0, 0]

        # count misclassifications and examples thru node
        self.misclassified = 0

        # validation error for subtree rooted at node
        self.error = 0

        self.train(data,labels)

    def train(self,data,labels):
        self.data = data
        self.labels = labels

        if self.tree.sampleFeatures == True:
            #sampleSize = np.random.randint(1,data.shape[1]) #This is not that great
            #sampleSize = int(np.log2(data.shape[1]))+1 #This is reasonable
            sampleSize = 17 # Cross validation shows this to be the best number
            featuresToUse = np.random.choice(self.tree.validFeatures,size=sampleSize,replace=False)
        else:
            featuresToUse = self.tree.validFeatures

        # If max depth was reached, stop splitting
        if self.tree.maxDepth and self.depth == self.tree.maxDepth:
            self.label = np.round(labels.sum()/labels.shape[0])
            return

        numPosLabels = labels.sum()
        # If all labels the same stop splitting
        if numPosLabels == labels.shape[0] or numPosLabels == 0:
            self.label = np.round(numPosLabels/labels.shape[0])
            return

        featureIndex = self.minImpurityFeature(data,labels,featuresToUse)
        # If entropy gain is zero stop splitting
        if featureIndex == None:
            self.label = np.round(numPosLabels/labels.shape[0])
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
                featureMatches = (feature == val)
                idx_val = np.where(featureMatches)
                p_val = float(featureMatches.sum())/labels.size
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

    def pruneClassify(self,point,label):
        if self.left and self.right:
            if point[self.featureIndex] == True:
                assignedlabel = self.left.pruneClassify(point, label)
            else:
                assignedlabel = self.right.pruneClassify(point, label)
        else:
            assignedlabel = self.label

        # count labels assigned and actual to prune after
        self.actualCount[assignedlabel] += 1

        # calc error over subtree rooted at node
        if assignedlabel != label:
            self.misclassified += 1
        self.error = self.misclassified / sum(self.actualCount)

        return assignedlabel

    def prune(self):
        # must be called AFTER pruneclassification!
        if self.left and self.right:
            if sum(self.actualCount):
                pruneError = min(self.actualCount) / sum(self.actualCount)
            else:
                pruneError = 1
            if pruneError < self.error:
                self.left = None
                self.right = None
                self.label = np.argmax(self.actualCount)
            else:
                self.left.prune()
                self.right.prune()
