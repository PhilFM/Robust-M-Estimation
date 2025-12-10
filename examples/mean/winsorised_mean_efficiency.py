import numpy as np
import random
import math

if __name__ == "__main__":
    import sys
    sys.path.append("../../pypi_package/src")

from winsorised_mean import winsorised_mean
from weighted_mean import weighted_mean

def main(test_run:bool, output_folder:str="../../Output"):
    sigma = 1.0
    n_samples = 3000
    small_val = 1.e-10

    n_array = [5,10,50,100]
    for n in n_array:
        tmstot = 0.0
        mstot = 0.0
        trim_size = 2*(n//5)
        for test_idx in range(n_samples):
            data = np.zeros(n)
            weight = np.zeros(n)
            for i in range(n):
                data[i] = random.gauss(0.0, sigma)
                weight[i] = 1.0

            mwinsorised = winsorised_mean(data, weight, trim_size=trim_size)
            tmstot += mwinsorised*mwinsorised

            lsm = weighted_mean(data, weight)
            mstot += lsm*lsm
        
        mlsvar = n*mstot/n_samples
        var = n*tmstot/n_samples
        if not test_run:
            print("n=",n," trim_size=",trim_size," efficiency=", (mlsvar+small_val)/(var+small_val))

    # test median
    n_array = [5,9,49,99,999]
    for n in n_array:
        tmstot = 0.0
        mstot = 0.0
        trim_size = n-1
        for test_idx in range(n_samples):
            data = np.zeros(n)
            weight = np.zeros(n)
            for i in range(n):
                data[i] = random.gauss(0.0, sigma)
                weight[i] = 1.0

            mwinsorised = winsorised_mean(data, weight, trim_size=trim_size)
            tmstot += mwinsorised*mwinsorised

            lsm = weighted_mean(data, weight)
            mstot += lsm*lsm
        
        mlsvar = n*mstot/n_samples
        var = n*tmstot/n_samples
        if not test_run:
            print("n=",n," median efficiency=", (mlsvar+small_val)/(var+small_val))

    if not test_run:
        print("Theoretical median limit=",2.0/math.pi)

    if test_run:
        print("winsorised_mean_efficiency OK")

if __name__ == "__main__":
    main(False) # test_run
