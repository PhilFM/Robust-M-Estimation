import numpy as np
import random
import math
import argparse

from winsorised_mean import winsorised_mean
from weighted_mean import weighted_mean

def main(testrun:bool):
    sigma = 1.0
    nSamples = 3000
    smallVal = 1.e-10

    Narray = [5,10,50,100]
    for N in Narray:
        tmstot = 0.0
        mstot = 0.0
        trimSize = 2*(N//5)
        for test_idx in range(nSamples):
            data = np.zeros(N)
            weight = np.zeros(N)
            for i in range(N):
                data[i] = random.gauss(0.0, sigma)
                weight[i] = 1.0

            mwinsorised = winsorised_mean(data, weight, trimSize=trimSize)
            tmstot += mwinsorised*mwinsorised

            lsm = weighted_mean(data, weight)
            mstot += lsm*lsm
        
        mlsvar = N*mstot/nSamples
        var = N*tmstot/nSamples
        if not testrun:
            print("N=",N," trimSize=",trimSize," efficiency=", (mlsvar+smallVal)/(var+smallVal))

    # test median
    Narray = [5,9,49,99,999]
    for N in Narray:
        tmstot = 0.0
        mstot = 0.0
        trimSize = N-1
        for test_idx in range(nSamples):
            data = np.zeros(N)
            weight = np.zeros(N)
            for i in range(N):
                data[i] = random.gauss(0.0, sigma)
                weight[i] = 1.0

            mwinsorised = winsorised_mean(data, weight, trimSize=trimSize)
            tmstot += mwinsorised*mwinsorised

            lsm = weighted_mean(data, weight)
            mstot += lsm*lsm
        
        mlsvar = N*mstot/nSamples
        var = N*tmstot/nSamples
        if not testrun:
            print("N=",N," median efficiency=", (mlsvar+smallVal)/(var+smallVal))

    if not testrun:
        print("Theoretical median limit=",2.0/math.pi)

    if testrun:
        print("OK")

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--testrun', action="store_true", default=False)
args = parser.parse_args()
main(args.testrun)
