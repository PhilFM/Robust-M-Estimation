import numpy as np
import math
import sys

sys.path.append("..")
from weightedMean import weightedMean

class WelschMean:
    def __init__(self, paramInstance, data, weight=None):
        self.paramInstance = paramInstance
        self.data = data
        if weight is None:
            self.weight = np.zeros(len(data))
            self.weight[:] = 1.0
        else:
            if len(weight) != len(data):
                raise ValueError("Inconsistent weight array")

            self.weight = weight

    # 1.0 means optimise for minimum, -1.0 means optimise for maximum
    def objectiveFuncSign(self):
        return -1.0

    def objectiveFunc(self, state, weight=None, stateRef=None):
        tot = 0.0
        invVar = 1.0/(self.paramInstance.sigma*self.paramInstance.sigma)
        m = state[0]
        if weight is None:
            weight = self.weight

        for d,w in zip(self.data,weight):
            diff = m-d
            tot += w*math.exp(-0.5*diff*diff*invVar)

        return tot

    def gradient(self, state, weight=None, stateRef=None):
        tot = 0.0
        invVar = 1.0/(self.paramInstance.sigma*self.paramInstance.sigma)
        m = state[0]
        if weight is None:
            weight = self.weight

        for d,w in zip(self.data,weight):
            diff = m-d
            tot -= w*diff*math.exp(-0.5*diff*diff*invVar)

        return [tot*invVar]

    # for checking algorithms
    def weightSum(self, state, weight=None):
        tot = 0.0
        invVar = 1.0/(self.paramInstance.sigma*self.paramInstance.sigma)
        m = state[0]
        if weight is None:
            weight = self.weight

        for d,w in zip(self.data,weight):
            diff = m-d
            tot -= w*math.exp(-0.5*diff*diff*invVar)

        return tot*invVar

    def secondDeriv(self, state, weight=None, stateRef=None):
        tot = 0.0
        invVar = 1.0/(self.paramInstance.sigma*self.paramInstance.sigma)
        m = state[0]
        if weight is None:
            weight = self.weight

        for d,w in zip(self.data,weight):
            diff = m-d
            tot += w*(diff*diff*invVar - 1.0)*math.exp(-0.5*diff*diff*invVar)

        return [[tot*invVar]]

    def weightedDeriv(self, state, lambdaVal: float, weight=None, stateRef=None):
        tot = 0.0
        invVar = 1.0/(self.paramInstance.sigma*self.paramInstance.sigma)
        m = state[0]
        if weight is None:
            weight = self.weight

        for d,w in zip(self.data,weight):
            diff = m-d
            tot += w*(lambdaVal*diff*diff*invVar - 1.0)*math.exp(-0.5*diff*diff*invVar)

        return [[tot*invVar]]

    # unused feature
    def calcStateRef(self, state, prevStateRef=None):
        return None

    def updateWeights(self, state, weight):
        invVar = 1.0/(self.paramInstance.sigma*self.paramInstance.sigma)
        m = state[0]
        for i,d in enumerate(self.data):
            diff = m-d
            fid = diff*diff*invVar
            w = math.exp(-0.5*fid)
            weight[i] = self.weight[i]*w

    def weightedFit(self, weight=None):
        if weight is None:
            weight = self.weight

        return weightedMean(self.data, weight)
