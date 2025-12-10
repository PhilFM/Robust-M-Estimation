import numpy as np
import random
import math

if __name__ == "__main__":
    import sys
    sys.path.append("../../pypi_package/src")

from trimmed_mean import trimmed_mean
from weighted_mean import weighted_mean

def main(test_run:bool, output_folder:str="../../Output"):
    sigma = 1.0
    nSamples = 3000
    smallVal = 1.e-10

    n_array = [5,10,50,100]
    for n in n_array:
        tmstot = 0.0
        mstot = 0.0
        trim_size = 2*(n//5)
        for test_idx in range(nSamples):
            data = np.zeros(n)
            weight = np.zeros(n)
            for i in range(n):
                data[i] = random.gauss(0.0, sigma)
                weight[i] = 1.0

            mtrimmed = trimmed_mean(data, weight, trim_size=trim_size)
            tmstot += mtrimmed*mtrimmed

            lsm = weighted_mean(data, weight)
            mstot += lsm*lsm
        
        mlsvar = n*mstot/nSamples
        var = n*tmstot/nSamples
        if not test_run:
            print("n=",n," trim_size=",trim_size," efficiency=", (mlsvar+smallVal)/(var+smallVal))

    # test median
    n_array = [5,9,49,99,999]
    for n in n_array:
        tmstot = 0.0
        mstot = 0.0
        trim_size = n-1
        for test_idx in range(nSamples):
            data = np.zeros(n)
            weight = np.zeros(n)
            for i in range(n):
                data[i] = random.gauss(0.0, sigma)
                weight[i] = 1.0

            mtrimmed = trimmed_mean(data, weight, trim_size=trim_size)
            tmstot += mtrimmed*mtrimmed

            lsm = weighted_mean(data, weight)
            mstot += lsm*lsm
        
        mlsvar = n*mstot/nSamples
        var = n*tmstot/nSamples
        if not test_run:
            print("n=",n," median efficiency=", (mlsvar+smallVal)/(var+smallVal))

    if not test_run:
        print("Theoretical median limit=",2.0/math.pi)

    if test_run:
        print("trimmed_mean_efficiency OK")

if __name__ == "__main__":
    main(False) # test_run
