import numpy as np
import math
import sys

sys.path.append("..")
from weightedMean import weightedMean

class WelschMean:
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

    # 1.0 means optimise for minimum, -1.0 means optimise for maximum
    def objectiveFuncSign(self):
        return -1.0

    def objectiveFunc(self, state, weight=None, stateRef=None):
        m = state[0]
        if weight is None:
            weight = self.weight

        tot = 0.0
        if self.scale is None:
            invVar = 1.0/(self.paramInstance.sigma*self.paramInstance.sigma)
            for d,w in zip(self.data,weight, strict=True):
                diff = m-d
                tot += w*math.exp(-0.5*diff*diff*invVar)
        else:
            for d,w,s in zip(self.data,weight,self.scale, strict=True):
                sigma = s*self.paramInstance.sigma
                diff = m-d
                tot += w*math.exp(-0.5*diff*diff/(sigma*sigma))

        return tot

    def gradient(self, state, weight=None, stateRef=None):
        m = state[0]
        if weight is None:
            weight = self.weight

        tot = 0.0
        if self.scale is None:
            invVar = 1.0/(self.paramInstance.sigma*self.paramInstance.sigma)
            for d,w in zip(self.data,weight, strict=True):
                diff = m-d
                tot -= w*diff*math.exp(-0.5*diff*diff*invVar)

            return [tot*invVar]
        else:
            for d,w,s in zip(self.data,weight,self.scale, strict=True):
                sigma = s*self.paramInstance.sigma
                invVar = 1.0/(sigma*sigma)
                diff = m-d
                tot -= w*invVar*diff*math.exp(-0.5*diff*diff*invVar)

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
                tot -= w*math.exp(-0.5*diff*diff*invVar)

            return tot*invVar
        else:
            for d,w,s in zip(self.data,weight,self.scale, strict=True):
                sigma = s*self.paramInstance.sigma
                invVar = 1.0/(sigma*sigma)
                diff = m-d
                tot -= w*invVar*math.exp(-0.5*diff*diff*invVar)

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
                tot += w*(diff*diff*invVar - 1.0)*math.exp(-0.5*diff*diff*invVar)

            return [[tot*invVar]]
        else:
            for d,w,s in zip(self.data,weight,self.scale, strict=True):
                sigma = s*self.paramInstance.sigma
                invVar = 1.0/(sigma*sigma)
                diff = m-d
                tot += w*invVar*(diff*diff*invVar - 1.0)*math.exp(-0.5*diff*diff*invVar)

            return [[tot]]

    def weightedDeriv(self, state, lambdaVal: float, weight=None, stateRef=None):
        m = state[0]
        if weight is None:
            weight = self.weight

        tot = 0.0
        if self.scale is None:
            invVar = 1.0/(self.paramInstance.sigma*self.paramInstance.sigma)
            for d,w in zip(self.data,weight, strict=True):
                diff = m-d
                tot += w*(lambdaVal*diff*diff*invVar - 1.0)*math.exp(-0.5*diff*diff*invVar)
                #print("m=",m,"d=",d,"w=",w,"diff=",diff,"fidp=",lambdaVal*diff*diff*invVar,"fid=",w*(lambdaVal*diff*diff*invVar - 1.0))#,"fid2=",math.exp(-0.5*diff*diff*invVar),"val=",w*(lambdaVal*diff*diff*invVar - 1.0)*math.exp(-0.5*diff*diff*invVar))

            return [[tot*invVar]]
        else:
            for d,w,s in zip(self.data,weight,self.scale, strict=True):
                sigma = s*self.paramInstance.sigma
                invVar = 1.0/(sigma*sigma)
                diff = m-d
                tot += w*invVar*(lambdaVal*diff*diff*invVar - 1.0)*math.exp(-0.5*diff*diff*invVar)

            return [[tot]]

    # unused feature
    def calcStateRef(self, state, prevStateRef=None):
        return None

    def updateWeights(self, state, weight):
        m = state[0]
        if self.scale is None:
            invVar = 1.0/(self.paramInstance.sigma*self.paramInstance.sigma)
            for i,d in enumerate(self.data):
                diff = m-d
                fid = diff*diff*invVar
                w = math.exp(-0.5*fid)
                weight[i] = self.weight[i]*w
        else:
            for i,(d,s) in enumerate(zip(self.data,self.scale, strict=True)):
                sigma = s*self.paramInstance.sigma
                invVar = 1.0/(sigma*sigma)
                diff = m-d
                fid = diff*diff*invVar
                w = math.exp(-0.5*fid)
                weight[i] = self.weight[i]*w

    def weightedFit(self, weight=None):
        if weight is None:
            weight = self.weight

        return weightedMean(self.data, weight, self.scale)
