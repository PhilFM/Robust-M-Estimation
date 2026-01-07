import numpy as np

from weighted_mean import weighted_mean

def winsorised_mean(data, weight, trim_size):
    # if array would be deleted by trim, return the median
    if 2*trim_size >= len(data):
        return np.median(data)

    # first sort the dat
    rdata = data.reshape(len(data))
    sorted_idx = rdata.argsort()

    winsorised_data = np.zeros((len(data),1))
    winsorised_weight = np.zeros(len(data))
    
    for i in range(trim_size,len(data)-trim_size):
        winsorised_data[i] = data[sorted_idx[i]]
        winsorised_weight[i] = weight[sorted_idx[i]]

    # now replace the trimmed data
    winsorised_data[0:trim_size] = data[sorted_idx[trim_size]]
    winsorised_data[len(data)-trim_size:len(data)] = data[sorted_idx[len(data)-trim_size-1]]
    winsorised_weight[0:trim_size] = weight[sorted_idx[trim_size]]
    winsorised_weight[len(data)-trim_size:len(data)] = weight[sorted_idx[len(data)-trim_size-1]]
    
    return weighted_mean(winsorised_data, winsorised_weight)

