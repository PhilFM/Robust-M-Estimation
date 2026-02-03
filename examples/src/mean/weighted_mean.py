import numpy as np
import numpy.typing as npt

def weighted_mean(data: npt.ArrayLike, weight: npt.ArrayLike=None, scale: npt.ArrayLike=None) -> np.array:
    totx = 0.0
    totw = 0.0
    if weight is None:
        for d in data:
            totx += d
            totw += 1.0
    elif scale is None:
        for d,w in zip(data,weight, strict=True):
            totx += w*d
            totw += w
    else:
        for d,w,s in zip(data,weight,scale, strict=True):
            w /= s*s
            totx += w*d
            totw += w

    return np.array(totx/totw)

