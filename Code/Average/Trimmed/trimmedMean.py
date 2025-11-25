import math
import numpy as np
import sys

sys.path.append("..")
from weightedMean import weightedMean

def trimmedMean(data, weight, trimSize):
    # if array would be deleted by trim, return the median
    if 2*trimSize >= len(data):
        return np.median(data)

    # first sort the data
    sortedData = data.argsort()
    #print("Sorted data: ", sortedData)

    # now remove the trimmed data
    trimmedData = sortedData[trimSize:len(data)-trimSize]
    #print("Trimmed data: ", trimmedData)
    
    return weightedMean(trimmedData, weight)

