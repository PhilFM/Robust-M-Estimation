import math
import numpy as np

from weighted_mean import weighted_mean

def winsorised_mean(data, weight, trimSize):
    # if array would be deleted by trim, return the median
    if 2*trimSize >= len(data):
        return np.median(data)

    # first sort the dat
    sortedData = data.argsort() #np.sort(data, key=lambda x: x[1])
    #print("trimSize=",trimSize," sorted data: ", sortedData)

    # now replace the trimmed data
    sortedData[0:trimSize] = sortedData[trimSize]
    sortedData[len(data)-trimSize:len(data)] = sortedData[len(data)-trimSize-1]
    #print("Winsorised data: ", sortedData)
    
    return weighted_mean(sortedData, weight)

