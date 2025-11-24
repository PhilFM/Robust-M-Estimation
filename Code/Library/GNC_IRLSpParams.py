import math

class GNC_IRLSpParams:
    def __init__(self, p, rscale, epsilonBase, epsilonLimit=None, beta=None):
        self.p = p
        self.rscale = rscale
        self.epsilonBase = epsilonBase
        self.epsilonLimit = epsilonBase if epsilonLimit is None else epsilonLimit
        self.beta = 1.0 if beta is None else beta

        # set parameters to final state
        self.reset(False)

    def reset(self, init=True):
        if init:
            self.epsilon = self.epsilonLimit
        else:
            self.epsilon = self.epsilonBase

    def finalStage(self):
        return True if self.epsilon <= self.epsilonBase else False

    def update(self):
        self.epsilon = max(self.epsilonBase, self.beta*math.pow(self.epsilon,2.0-self.p)) # epsilon = max(epsilonBase, beta*math.pow(epsilon,2.0-p))

    def summary(self):
        return "epsilon=" + str(self.epsilon)
    
