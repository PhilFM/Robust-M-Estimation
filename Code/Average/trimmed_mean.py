import math
import numpy as np
import sys

sys.path.append("..")
from weighted_mean import weighted_mean

def trimmed_mean(data, weight, trimSize):
    # if array would be deleted by trim, return the median
    if 2*trimSize >= len(data):
        return np.median(data)

    # first sort the data
    rdata = data.reshape(len(data))
    sortedIdx = rdata.argsort()
    #print("data: ", data)
    #print("Sorted indices: ", sortedIdx)

    # now remove the trimmed data
    trimmedIdx = sortedIdx[trimSize:len(data)-trimSize]
    #print("Trimmed indices: ", trimmedIdx)

    trimmedData = np.zeros((len(trimmedIdx),1))
    trimmedWeight = np.zeros(len(trimmedIdx))
    for i,idx in enumerate(trimmedIdx):
        trimmedData[i] = data[idx]
        trimmedWeight[i] = weight[idx]

    #print("Trimmed data=", trimmedData)
    return weighted_mean(trimmedData, trimmedWeight)

