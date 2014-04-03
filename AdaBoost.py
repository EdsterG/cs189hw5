from scipy import stats
from math import log, exp
from utils import *

class AdaBoost:
    def __init__(self, data, y, Classifier, params):
        # params is list of parameters for given Classifier
        self.Classifier = Classifier

        # hyps : list of trained classifiers
        self.hyps = []

        # hWeights : weight given to each hypothesis in vote (alpha)
        self.hWeights = []

        # dWeights : weights for each data entry (Dt)
        self.dWeights = np.ones(y.shape) * 1/np.len(y)

        self.train(data, y, params)

    def train(self, data, y, params):
        # train each weak learner using Dt
        for t in range(params):

            # sample data based on dWeights
            sampleInds = nSample(self.dWeights, range(y.shape[0]), y.shape[0])
            sData = data[sampleInds,:]

            self.hyp[t] = Classifier(sData, y, params[t])

            #calculate hypothesis weight
            pred = self.hyp[t].classify(sData)
            error = (pred != y).sum() / float(y.size)
            hWeights[t] = .5 * log((1-error) / error)

            # update D
            self.dWeights *= exp(-hWeights[t] * y * pred)
            self.dWeights /= la.norm(self.dWeights, 1)


    def classify(self, data):
        rez = np.zeros(data.shape[0])

        # sum alphas * hypothesis classifications
        for hi in len(self.hyps):
            rez += self.hWeights[hi] * self.hyps[hi].classify(data)
        
        # round to take vote
        return rez > .5

