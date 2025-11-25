import numpy as np
import random
import math
import sys

from trimmedMean import trimmedMean

sys.path.append("..")
from weightedMean import weightedMean

sigma = 1.0
nSamples = 3000
smallVal = 1.e-10

Narray = [5,10,50,100]
for N in Narray:
    tmstot = 0.0
    mstot = 0.0
    trimSize = 2*(N//5)
    for testIdx in range(nSamples):
        data = np.zeros(N)
        weight = np.zeros(N)
        for i in range(N):
            data[i] = random.gauss(0.0, sigma)
            weight[i] = 1.0

        mtrimmed = trimmedMean(data, weight, trimSize=trimSize)
        tmstot += mtrimmed*mtrimmed

        lsm = weightedMean(data, weight)
        mstot += lsm*lsm
        
    mlsvar = N*mstot/nSamples
    var = N*tmstot/nSamples
    print("N=",N," trimSize=",trimSize," efficiency=", (mlsvar+smallVal)/(var+smallVal))

# test median
Narray = [5,9,49,99,999]
for N in Narray:
    tmstot = 0.0
    mstot = 0.0
    trimSize = N-1
    for testIdx in range(nSamples):
        data = np.zeros(N)
        weight = np.zeros(N)
        for i in range(N):
            data[i] = random.gauss(0.0, sigma)
            weight[i] = 1.0

        mtrimmed = trimmedMean(data, weight, trimSize=trimSize)
        tmstot += mtrimmed*mtrimmed

        lsm = weightedMean(data, weight)
        mstot += lsm*lsm
        
    mlsvar = N*mstot/nSamples
    var = N*tmstot/nSamples
    print("N=",N," median efficiency=", (mlsvar+smallVal)/(var+smallVal))

print("Theoretical median limit=",2.0/math.pi)
