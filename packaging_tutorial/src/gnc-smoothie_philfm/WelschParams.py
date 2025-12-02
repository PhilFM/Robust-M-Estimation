import math

class WelschParams:
    def __init__(self, sigma):
        self.sigma = sigma

        # set parameters to final state
        self.reset(False)

    def reset(self, init=True):
        pass

    def finalStage(self):
        return True

    def update(self):
        pass

    def summary(self):
        return "sigma=" + str(self.sigma)
