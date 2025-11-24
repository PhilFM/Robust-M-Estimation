import numpy as np

def weightedMean(data, weight):
    totx = 0.0
    totw = 0.0
    for d,w in zip(data,weight):
        #print("Weighted mean d:",d)
        totx += w*d
        totw += w

    return np.array([totx/totw])

