import numpy as np

def weightedMean(data, weight, scale=None):
    totx = 0.0
    totw = 0.0
    if scale is None:
        for d,w in zip(data,weight, strict=True):
            totx += w*d
            totw += w

        return np.array([totx/totw])
    else:
        for d,w,s in zip(data,weight,scale, strict=True):
            w /= s*s
            totx += w*d
            totw += w

    return np.array([totx/totw])

