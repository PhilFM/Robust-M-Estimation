import math
import numpy as np

class GNC_IRLSpMean:
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
            for d,w in zip(self.data,weight, strict=True):
                r = rscale*math.fabs(m-d)
                if r < epsilon:
                    tot += w*0.5*math.pow(epsilon, p-2.0)*r*r/(rscale*rscale)
                else:
                    tot += w*(K + math.log(r)/(rscale*rscale))
        else:
            K = 0.5*math.pow(epsilon,p)*math.pow(rscale,-2.0) - math.pow(epsilon,p)*math.pow(rscale,-2.0)/p
            for d,w in zip(self.data,weight, strict=True):
                r = rscale*math.fabs(m-d)
                if r < epsilon:
                    tot += w*0.5*math.pow(epsilon, p-2.0)*r*r/(rscale*rscale)
                else:
                    tot += w*(K + math.pow(r,p)/(rscale*rscale)/p)

        return tot
