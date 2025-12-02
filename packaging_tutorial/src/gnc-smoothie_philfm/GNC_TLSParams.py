import math

class GNC_TLSParams:
    def __init__(self, c, rscale, muBase, muLimit=None, gamma=None):
        self.c = c
        self.rscale = rscale
        self.muBase = muBase
        self.muLimit = muBase if muLimit is None else muLimit
        self.gamma = 1.0 if gamma is None else gamma

        # set parameters to final state
        self.reset(False)

    def reset(self, init=True):
        if init:
            self.mu = self.muLimit
        else:
            self.mu = self.muBase

    def finalStage(self):
        return True if self.mu >= self.muBase else False

    def update(self):
        self.mu = min(self.muBase, self.gamma*math.sqrt(self.mu) if self.mu <= 1.0 else self.gamma*self.mu)

    def summary(self):
        return "mu=" + str(self.mu)
