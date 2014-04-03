from scipy import stats
from utils import *

class AdaBoost:
    def __init__(self, data, y, Classifier, params):
        # params is list of parameters for given Classifier
        self.Classifier = Classifier

        # hyps : list of trained classifiers
        self.hyps = []

        # hWeights : weight given to each hypothesis in vote (alpha)
        self.hWeights = np.zeros((len(params),1))

        # dWeights : weights for each data entry (Dt)
        self.dWeights = np.ones(y.shape) * 1/y.shape[0]

        self.train(data, y, params)

    def train(self, data, y, params):
        # train each weak learner using Dt
        for t in range(len(params)):
            print "training tree %d" % t

            # sample data based on dWeights
            sampleInds = nSample(self.dWeights, range(y.shape[0]), y.shape[0]/4)
            sData = data[sampleInds,:]
            sy = y[sampleInds,:]

            self.hyps.append(self.Classifier(sData, sy)) # params[t]

            #calculate hypothesis weight
            pred = self.hyps[t].classify(data)
            error = ((pred != y)*self.dWeights).sum()
            self.hWeights[t]=(.5 * np.log((1-error) / error))
            #self.hWeights /= la.norm(self.hWeights, 1)

            # update D
            self.dWeights *= np.exp(-self.hWeights[t] * y * pred)
            self.dWeights /= la.norm(self.dWeights, 1)


    def classify(self, data):
        reslut = np.zeros(data.shape)

        # sum alphas * hypothesis classifications
        for hi in range(len(self.hyps)):
            reslut += self.hWeights[hi] * self.hyps[hi].classify(data)
        
        # round to take vote
        return reslut > .5

