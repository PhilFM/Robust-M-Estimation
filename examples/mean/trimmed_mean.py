import numpy as np

from weighted_mean import weighted_mean

def trimmed_mean(data, weight, trim_size):
    # if array would be deleted by trim, return the median
    if 2*trim_size >= len(data):
        return np.median(data)

    # first sort the data
    rdata = data.reshape(len(data))
    sorted_idx = rdata.argsort()
    #print("data: ", data)
    #print("Sorted indices: ", sorted_idx)

    # now remove the trimmed data
    trimmed_idx = sorted_idx[trim_size:len(data)-trim_size]
    #print("Trimmed indices: ", trimmed_idx)

    trimmed_data = np.zeros((len(trimmed_idx),1))
    trimmed_weight = np.zeros(len(trimmed_idx))
    for i,idx in enumerate(trimmed_idx):
        trimmed_data[i] = data[idx]
        trimmed_weight[i] = weight[idx]

    #print("Trimmed data=", trimmed_data)
    return weighted_mean(trimmed_data, trimmed_weight)

