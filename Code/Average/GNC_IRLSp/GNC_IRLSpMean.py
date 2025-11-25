import math
import numpy as np
import sys

sys.path.append("..")
from weightedMean import weightedMean

def weightFunc(m, p, epsilon, rscale, d):
    r = rscale*math.fabs(m-d)
    return math.pow(max(r,epsilon), p-2.0)

class GNC_IRLSpMean:
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
        return 1.0

    def objectiveFunc(self, state, weight=None, stateRef=None):
        # we have d(rho)/d(x) 
        #         ----------- = (max(rscale*|x-d|,epsilon))^(p-2)
        #             x-d
        # So
        #    d(rho)   
        #    ------ = (x-d)(max(rscale*|x-d|,epsilon))^(p-2)
        #     d(x) 
        #           = epsilon^(p-2)*(x-d)                if rscale*|x-d| < epsilon
        #             sign(x-d)*|x-d|^(p-1)*rscale^(p-2) otherwise
        #
        # So rho(x) = (rscale*epsilon)^(p-2)*(x-d)^2/2           if rscale*|x-d| < epsilon
        #             K + rscale^-2*log(rscale*|x-d|)  if rscale*|x-d| >= epsilon and p=0
        #             K + rscale^(p-2)*|x-d|^p/p       otherwise
        #
        # where K = (rscale*epsilon)^(p-2)*(epsilon/rscale)^2/2 - rscale^-2*log(epsilon)            if p = 0
        #       K = (rscale*epsilon)^(p-2)*(epsilon/rscale)^2/2 - rscale^(p-2)*(epsilon/rscale)^p/p if p != 0
        # or
        #
        # where K = epsilon^p*rscale^(p-4)/2 - rscale^-2*log(epsilon)    if p = 0
        #       K = epsilon^p*rscale^(p-4)/2 - epsilon^p*rscale^-2/p     if p != 0
        m = state[0]
        p = self.paramInstance.p
        rscale = self.paramInstance.rscale
        epsilon = self.paramInstance.epsilon
        if weight is None:
            weight = self.weight

        tot = 0.0
        if p == 0.0:
            K = 0.5*math.pow(epsilon,p)*math.pow(rscale,p-2.0) - math.pow(rscale,-2.0)*math.log(epsilon)
            #print("K=",K,0.5*math.pow(epsilon,p)*math.pow(rscale,p-2.0),math.pow(rscale,-2.0)*math.log(epsilon))
            for d in self.data:
                sgn = np.sign(m-d)
                r = rscale*math.fabs(m-d)
                if r < epsilon:
                    tot += 0.5*math.pow(epsilon, p-2.0)*r*r/(rscale*rscale)
                else:
                    tot += K + math.log(r)/(rscale*rscale)
        else:
            K = 0.5*math.pow(epsilon,p)*math.pow(rscale,-2.0) - math.pow(epsilon,p)*math.pow(rscale,-2.0)/p
            for d in self.data:
                sgn = np.sign(m-d)
                r = rscale*math.fabs(m-d)
                if r < epsilon:
                    tot += 0.5*math.pow(epsilon, p-2.0)*r*r/(rscale*rscale)
                else:
                    tot += K + math.pow(r,p)/(rscale*rscale)/p

        return tot

    def gradient(self, state, weight=None, stateRef=None):
        m = state[0]
        p = self.paramInstance.p
        rscale = self.paramInstance.rscale
        epsilon = self.paramInstance.epsilon
        if weight is None:
            weight = self.weight

        tot = 0.0
        for d in self.data:
            tot += (m-d)*weightFunc(m,p,epsilon,rscale,d)

        return [tot]

    # for checking algorithms
    def weightSum(self, state, weight=None):
        m = state[0]
        p = self.paramInstance.p
        rscale = self.paramInstance.rscale
        epsilon = self.paramInstance.epsilon
        if weight is None:
            weight = self.weight

        tot = 0.0
        for d in self.data:
            tot += weightFunc(m,p,epsilon,rscale,d)

        return tot

    def secondDeriv(self, state, weight=None, stateRef=None):
        smallDiff = 1.e-5
        if weight is None:
            weight = self.weight

        return [[0.5*(self.gradient([state[0]+smallDiff], weight)[0]
                      - self.gradient([state[0]-smallDiff], weight)[0])/smallDiff]]

    def weightedDeriv(self, state, lambdaVal: float, weight=None, stateRef=None):
        if weight is None:
            weight = self.weight

        return [[(1.0-lambdaVal)*self.weightSum(state,weight) + lambdaVal*self.secondDeriv(state,weight)[0][0]]]

    # unused feature
    def calcStateRef(self, state, prevStateRef=None):
        return None

    def updateWeights(self, state, weight):
        m = state[0]
        p = self.paramInstance.p
        rscale = self.paramInstance.rscale
        epsilon = self.paramInstance.epsilon
        for i,d in enumerate(self.data):
            weight[i] = self.weight[i]*weightFunc(m,p,epsilon,rscale,d)

    def weightedFit(self, weight=None):
        if weight is None:
            weight = self.weight

        return weightedMean(self.data, weight)
