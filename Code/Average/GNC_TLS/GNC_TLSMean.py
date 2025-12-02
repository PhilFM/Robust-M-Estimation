import math
import numpy as np
import sys

sys.path.append("..")
from weightedMean import weightedMean

def weightFunc(x, c, mu, rscale, d):
    r = rscale*math.fabs(x-d)
    if r <= c:
        #print("TLS-W small")
        return 1.0
    elif r*mu >= (1.0+mu)*c:
        #print("TLS-W large")
        return 0.0
    else:
        #print("TLS-W just right")
        return c*(1.0+mu)/r - mu

class GNC_TLSMean:
    def __init__(self, paramInstance, data, weight=None):
        self.paramInstance = paramInstance
        self.data = data
        self.weight = weight
        if weight is None:
            self.weight = np.zeros(len(data))
            self.weight[:] = 1.0
        else:
            if len(weight) != len(data):
                raise ValueError("Inconsistent weight array")

    # 1.0 means optimise for minimum, -1.0 means optimise for maximum
    def objectiveFuncSign(self):
        return 1.0

    def objectiveFunc(self, state, weight=None, stateRef=None):
        # we have d(rho)/d(x) 
        #         ----------- = c*(1+mu)/|x-d| - mu   for 0 <= |x-d| <= c*(1+mu)
        #             x-d
        # So
        #    d(rho)   
        #    ------ = (x-d)(c*(1+mu)/|x-d| - mu)
        #     d(x) 
        #           = c*(1+mu)*sign(x-d) - mu*(x-d)
        #
        # So rho(x) = c*(1+mu)*|x-d| - mu*(x-d)^2/2 + C for some constant C
        # For |x-d| <= c we have d(rho)/dx = x-d thus rho(x) = (x-d)^2/2, so C = c^2/2 - c*(1+mu)*c + mu*c^2/2 = -(1+mu)*c^2/2
        m = state[0]
        c = self.paramInstance.c
        rscale = self.paramInstance.rscale
        mu = self.paramInstance.mu
        C = -0.5*(1.0+mu)*c*c
        rlimit = (1.0+mu)*c/mu
        #print("rlimit=",rlimit,"rscale=",rscale,"C=",C)
        if weight is None:
            weight = self.weight

        tot = 0.0
        for d,w in zip(self.data,weight, strict=True):
            r = rscale*math.fabs(m-d)
            if r <= c:
                #print("TLS small")
                tot += w*0.5*r*r/(rscale*rscale)
            elif r >= rlimit:
                #print("TLS large")
                tot += w*(0.5*c*c*(1.0+mu)/mu)/(rscale*rscale)
            else:
                #print("TLS just right")
                tot += w*(c*(1.0+mu)*r - 0.5*mu*r*r + C)/(rscale*rscale)

        return tot

    def gradient(self, state, weight=None, stateRef=None):
        m = state[0]
        c = self.paramInstance.c
        rscale = self.paramInstance.rscale
        mu = self.paramInstance.mu
        if weight is None:
            weight = self.weight

        tot = 0.0
        for d,w in zip(self.data,weight, strict=True):
            tot += w*(m-d)*weightFunc(m,c,mu,rscale,d)

        return [tot]

    # for checking algorithms
    def weightSum(self, state, weight=None):
        m = state[0]
        c = self.paramInstance.c
        rscale = self.paramInstance.rscale
        mu = self.paramInstance.mu
        if weight is None:
            weight = self.weight

        tot = 0.0
        for d,w in zip(self.data,weight, strict=True):
            tot += w*weightFunc(m,c,mu,rscale,d)

        return tot

    def secondDeriv(self, state, weight=None, stateRef=None):
        m = state[0]
        c = self.paramInstance.c
        rscale = self.paramInstance.rscale
        mu = self.paramInstance.mu
        smallDiff = 1.e-5
        if weight is None:
            weight = self.weight

        return [[0.5*(self.gradient([m+smallDiff], weight)[0]
                      - self.gradient([m-smallDiff], weight)[0])/smallDiff]]

    def weightedDeriv(self, state, lambdaVal: float, weight=None, stateRef=None):
        if weight is None:
            weight = self.weight

        return [[(1.0-lambdaVal)*self.weightSum(state,weight) + lambdaVal*self.secondDeriv(state,weight)[0][0]]]

    # unused feature
    def calcStateRef(self, state, prevStateRef=None):
        return None

    def updateWeights(self, state, weight):
        m = state[0]
        c = self.paramInstance.c
        rscale = self.paramInstance.rscale
        mu = self.paramInstance.mu
        for i,d in enumerate(self.data):
            weight[i] = self.weight[i]*weightFunc(m,c,mu,rscale,d)

    def weightedFit(self, weight=None):
        if weight is None:
            weight = self.weight

        return weightedMean(self.data, weight)
