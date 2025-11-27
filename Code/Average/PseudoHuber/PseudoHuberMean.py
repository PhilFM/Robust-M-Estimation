import numpy as np
import math
import sys

sys.path.append("..")
from weightedMean import weightedMean

class PseudoHuberMean:
    def __init__(self, paramInstance, data, weight=None, scale=None):
        self.paramInstance = paramInstance
        self.data = data
        if weight is None:
            self.weight = np.zeros(len(data))
            self.weight[:] = 1.0
        else:
            if len(weight) != len(data):
                raise ValueError("Inconsistent weight array")

            self.weight = weight

        self.scale = scale

    def objectiveFuncSign(self):
        return 1.0

    def objectiveFunc(self, state, weight=None, stateRef=None):
        m = state[0]
        if weight is None:
            weight = self.weight

        tot = 0.0
        if self.scale is None:
            invVar = 1.0/(self.paramInstance.sigma*self.paramInstance.sigma)
            for d,w in zip(self.data,weight, strict=True):
                diff = m-d
                tot += w*(math.sqrt(1.0 + diff*diff*invVar) - 1.0)

            return tot*self.paramInstance.sigma*self.paramInstance.sigma
        else:
            for d,w,s in zip(self.data,weight,self.scale, strict=True):
                sigma = s*self.paramInstance.sigma
                diff = m-d
                tot += w*sigma*sigma*(math.sqrt(1.0 + diff*diff/(sigma*sigma)) - 1.0)

            return tot

    def gradient(self, state, weight=None, stateRef=None):
        tot = 0.0
        m = state[0]
        if weight is None:
            weight = self.weight

        if self.scale is None:
            invVar = 1.0/(self.paramInstance.sigma*self.paramInstance.sigma)
            for d,w in zip(self.data,weight, strict=True):
                diff = m-d
                tot += w*diff/math.sqrt(1.0 + diff*diff*invVar)
        else:
            for d,w,s in zip(self.data,weight,self.scale, strict=True):
                sigma = s*self.paramInstance.sigma
                diff = m-d
                tot += w*diff/math.sqrt(1.0 + diff*diff/(sigma*sigma))

        return [tot]

    # for checking algorithms
    def weightSum(self, state, weight=None):
        m = state[0]
        if weight is None:
            weight = self.weight

        tot = 0.0
        if self.scale is None:
            invVar = 1.0/(self.paramInstance.sigma*self.paramInstance.sigma)
            for d,w in zip(self.data,weight, strict=True):
                diff = m-d
                tot += w/math.sqrt(1.0 + diff*diff*invVar)
        else:
            for d,w,s in zip(self.data,weight,self.scale, strict=True):
                sigma = s*self.paramInstance.sigma
                diff = m-d
                tot += w/math.sqrt(1.0 + diff*diff/(sigma*sigma))

        return tot

    def secondDeriv(self, state, weight=None, stateRef=None):
        m = state[0]
        if weight is None:
            weight = self.weight

        tot = 0.0
        if self.scale is None:
            invVar = 1.0/(self.paramInstance.sigma*self.paramInstance.sigma)
            for d,w in zip(self.data,weight, strict=True):
                diff = m-d
                fid = math.sqrt(1.0 + diff*diff*invVar)
                tot += w*(1.0 - diff*diff*invVar/(fid*fid))/fid
        else:
            for d,w,s in zip(self.data,weight,self.scale, strict=True):
                sigma = s*self.paramInstance.sigma
                invVar = 1.0/(sigma*sigma)
                fid = math.sqrt(1.0 + diff*diff*invVar)
                tot += w*(1.0 - diff*diff*invVar/(fid*fid))/fid

        return [[tot]]

    def weightedDeriv(self, state, lambdaVal, weight=None, stateRef=None):
        m = state[0]
        if weight is None:
            weight = self.weight

        tot = 0.0
        if self.scale is None:
            invVar = 1.0/(self.paramInstance.sigma*self.paramInstance.sigma)
            for d,w in zip(self.data,weight, strict=True):
                diff = m-d
                fid = math.sqrt(1.0 + diff*diff*invVar)
                tot += w*(1.0 - lambdaVal*diff*diff*invVar/(fid*fid))/fid
        else:
            for d,w,s in zip(self.data,weight,self.scale, strict=True):
                sigma = s*self.paramInstance.sigma
                invVar = 1.0/(sigma*sigma)
                diff = m-d
                fid = math.sqrt(1.0 + diff*diff*invVar)
                tot += w*(1.0 - lambdaVal*diff*diff*invVar/(fid*fid))/fid

        return [[tot]]

    # unused feature
    def calcStateRef(self, state, prevStateRef=None):
        return None

    def updateWeights(self, state, weight: np.array):
        m = state[0]
        if self.scale is None:
            invVar = 1.0/(self.paramInstance.sigma*self.paramInstance.sigma)
            for i,d in enumerate(self.data):
                diff = m-d
                weight[i] = self.weight[i]/math.sqrt(1.0 + diff*diff*invVar)
        else:
            for i,(d,s) in enumerate(zip(self.data,self.scale, strict=True)):
                sigma = s*self.paramInstance.sigma
                invVar = 1.0/(sigma*sigma)
                diff = m-d
                weight[i] = self.weight[i]/math.sqrt(1.0 + diff*diff*invVar)
        
    def weightedFit(self, weight=None):
        if weight is None:
            weight = self.weight

        return weightedMean(self.data, weight, self.scale)
