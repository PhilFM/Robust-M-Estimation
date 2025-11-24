import math

class GNC_WelschParams:
    def __init__(self, sigmaBase, sigmaLimit=None, numSigmaSteps=None, maxNIterations: int=1000000):
        self.sigmaBase = sigmaBase
        self.sigmaLimit = sigmaBase if sigmaLimit is None else sigmaLimit
        if self.sigmaLimit != self.sigmaBase:
            # GNC schedule for sigma
            self.numSigmaSteps = maxNIterations if numSigmaSteps is None else numSigmaSteps
            if self.numSigmaSteps > maxNIterations:
                raise ValueError("Too many sigma steps")

            self.beta = math.exp((math.log(sigmaBase) - math.log(self.sigmaLimit))/(self.numSigmaSteps - 1.0))
        else:
            # fixed sigma
            self.numSigmaSteps = 0
            self.beta = 1.0

        # set parameters to final state
        self.reset(False)

    def reset(self, init=True):
        if init:
            self.sigma = self.sigmaLimit
        else:
            self.sigma = self.sigmaBase

    def finalStage(self):
        return True if self.sigma <= self.sigmaBase else False

    def update(self):
        self.sigma = max(self.sigmaBase, self.beta*self.sigma)

    def summary(self):
        return "sigma=" + str(self.sigma)
