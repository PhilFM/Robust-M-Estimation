import numpy as np
import math
import sys

sys.path.append("..")
from weightedMean import weightedMean

class GemanMcClureMean:
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

    #               d^2
    # rho(x) = -------------
    #          sigma^2 + d^2
    #
    #  where d = x - z
    def objectiveFunc(self, state, weight=None, stateRef=None):
        m = state[0]
        if weight is None:
            weight = self.weight

        tot = 0.0
        if self.scale is None:
            var = self.paramInstance.sigma*self.paramInstance.sigma
            for d,w in zip(self.data,weight):
                diff = m-d
                diffSqr = diff*diff
                tot += w*diffSqr/(var + diffSqr)
        else:
            for d,w,s in zip(self.data,weight,self.scale, strict=True):
                sigma = s*self.paramInstance.sigma
                diff = m-d
                diffSqr = diff*diff
                tot += w*diffSqr/(sigma*sigma + diffSqr)

        return tot

    # d(rho)   d(rho)   2*(sigma^2 + d^2)*d - 2*d^3
    # ------ = ------ = ----------------------------
    #  d(x)     d(d)       (sigma^2 + d^2)^2
    #
    #             2*sigma^2*d
    #        = -----------------
    #          (sigma^2 + d^2)^2
    def gradient(self, state, weight=None, stateRef=None):
        m = state[0]
        if weight is None:
            weight = self.weight

        tot = 0.0
        if self.scale is None:
            var = self.paramInstance.sigma*self.paramInstance.sigma
            for d,w in zip(self.data,weight):
                diff = m-d
                diffSqr = diff*diff
                v = var + diffSqr
                tot += w*2.0*var*diff/(v*v)
        else:
            for d,w,s in zip(self.data,weight,self.scale, strict=True):
                sigma = s*self.paramInstance.sigma
                var = sigma*sigma
                diff = m-d
                diffSqr = diff*diff
                v = var + diffSqr
                tot += w*2.0*var*diff/(v*v)

        return [tot]

    # for checking algorithms
    def weightSum(self, state, weight=None):
        m = state[0]
        if weight is None:
            weight = self.weight

        tot = 0.0
        if self.scale is None:
            var = self.paramInstance.sigma*self.paramInstance.sigma
            for d,w in zip(self.data,weight):
                diff = m-d
                diffSqr = diff*diff
                v = var + diffSqr
                tot += w*2.0*var/(v*v)
        else:
            for d,w,s in zip(self.data,weight,self.scale, strict=True):
                sigma = s*self.paramInstance.sigma
                var = sigma*sigma
                diff = m-d
                diffSqr = diff*diff
                v = var + diffSqr
                tot += w*2.0*var/(v*v)

        return tot

    # d^2(rho)   2*sigma^2*[(sigma^2 + d^2)^2 - 4*(sigma^2 + d^2)*d^2]
    # -------- = ----------------------------------------------------
    #  d(x)^2                (sigma^2 + d^2)^4
    #
    #            2*sigma^2*(sigma^2 - 3*d^2)
    #          = ---------------------------
    #                (sigma^2 + d^2)^3
    def secondDeriv(self, state, weight=None, stateRef=None):
        m = state[0]
        if weight is None:
            weight = self.weight

        tot = 0.0
        if self.scale is None:
            var = self.paramInstance.sigma*self.paramInstance.sigma
            for d,w in zip(self.data,weight):
                diff = m-d
                diffSqr = diff*diff
                v = var + diffSqr
                tot += w*2.0*var*(var - 3.0*diffSqr)/(v*v*v)
        else:
            for d,w,s in zip(self.data,weight,self.scale, strict=True):
                sigma = s*self.paramInstance.sigma
                var = sigma*sigma
                diff = m-d
                diffSqr = diff*diff
                v = var + diffSqr
                tot += w*2.0*var*(var - 3.0*diffSqr)/(v*v*v)

        return [[tot]]

    def weightedDeriv(self, state, lambdaVal, weight=None, stateRef=None):
        m = state[0]
        if weight is None:
            weight = self.weight

        tot = 0.0
        if self.scale is None:
            var = self.paramInstance.sigma*self.paramInstance.sigma
            for d,w in zip(self.data,weight):
                diff = m-d
                diffSqr = diff*diff
                v = var + diffSqr
                tot += w*2.0*var*(1.0 + lambdaVal*((var - 3.0*diffSqr)/v - 1.0))/(v*v)
        else:
            for d,w,s in zip(self.data,weight,self.scale, strict=True):
                sigma = s*self.paramInstance.sigma
                var = sigma*sigma
                diff = m-d
                diffSqr = diff*diff
                v = var + diffSqr
                tot += w*2.0*var*(1.0 + lambdaVal*((var - 3.0*diffSqr)/v - 1.0))/(v*v)

        return [[tot]]

    # unused feature
    def calcStateRef(self, state, prevStateRef=None):
        return None

    def updateWeights(self, state, weight: np.array):
        m = state[0]
        if self.scale is None:
            var = self.paramInstance.sigma*self.paramInstance.sigma
            for i,d in enumerate(self.data):
                diff = m-d
                diffSqr = diff*diff
                v = var + diffSqr
                weight[i] = self.weight[i]*2.0*var/(v*v)
        else:
            for i,(d,s) in enumerate(zip(self.data,self.scale, strict=True)):
                sigma = s*self.paramInstance.sigma
                var = sigma*sigma
                diff = m-d
                diffSqr = diff*diff
                v = var + diffSqr
                weight[i] = self.weight[i]*2.0*var/(v*v)

    def weightedFit(self, weight=None):
        if weight is None:
            weight = self.weight

        return weightedMean(self.data, weight, self.scale)
