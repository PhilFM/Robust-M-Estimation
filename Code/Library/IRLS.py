import math
import numpy as np

class IRLS:
    def __init__(self, algInstance, 
                 maxNIterations=50, diffThres=1.e-12, printWarnings=False, stateStart=None, debug=False):
        self.algInstance = algInstance
        self.maxNIterations = maxNIterations
        self.diffThres = diffThres
        self.printWarnings = printWarnings
        self.stateStart = stateStart
        self.debug = debug

    def run(self):
        self.algInstance.paramInstance.reset()
        weight = np.copy(self.algInstance.weight)
        if self.stateStart is None:
            state = self.algInstance.weightedFit(weight)
        else:
            state = self.stateStart

        if self.printWarnings:
            print("Initial state=",state,"params=",self.algInstance.paramInstance.summary(),"diffThres=",self.diffThres)

        if self.debug:
            diffs = []
            stateList = []

        for itn in range(self.maxNIterations):
            self.algInstance.updateWeights(state, weight)
            stateOld = state
            state = self.algInstance.weightedFit(weight)

            if self.algInstance.paramInstance.finalStage():
                if self.diffThres is not None:
                    stateMaxDiff = np.linalg.norm(state-stateOld, ord=np.inf)
                    if self.printWarnings:
                        print("stateMaxDiff=",stateMaxDiff)

                    if self.debug is True and stateMaxDiff > 0.0:
                        print("Adding diff stateMaxDiff", stateMaxDiff)
                        diffs.append(math.log10(stateMaxDiff))

                    if stateMaxDiff < self.diffThres:
                        if self.printWarnings:
                            print("Difference threshold reached")

                        break

            if self.printWarnings:
                print("itn=",itn,"state=",state,"params=",self.algInstance.paramInstance.summary())

            self.algInstance.paramInstance.update()
            if self.debug:
                stateList.append((itn/(self.maxNIterations-1), # alpha
                                  np.copy(state)))

        self.algInstance.paramInstance.reset(False) # finish with parameters in correct final state
        if self.debug:
            return state,itn+1,diffs,stateList
        else:
            return state
