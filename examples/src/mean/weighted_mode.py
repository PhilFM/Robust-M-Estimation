import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

def weighted_histogram(data:npt.ArrayLike, bin_size:float, weight:npt.ArrayLike=None, scale:npt.ArrayLike=None) -> tuple[float,np.array]:
    if scale is not None:
        weight = np.copy(weight)
        weight /= scale*scale

    data = data.reshape(len(data))
    x_min = min(data)
    x_max = max(data)
    x_range = x_max-x_min
    n_bins = 1+int(x_range/bin_size)
    if n_bins == 0:
        return 0.5*(x_min + x_max)

    range1 = (x_min,x_min+n_bins*bin_size)
    range2 = (x_min+0.5*bin_size,x_min+(0.5+n_bins)*bin_size)
    if weight is None:
        counts1,bins1 = np.histogram([data], bins=n_bins, range=range1)
        counts2,bins2 = np.histogram([data], bins=n_bins, range=range2)
    else:
        counts1,bins1 = np.histogram([data], bins=n_bins, weights=[weight], range=range1)
        counts2,bins2 = np.histogram([data], bins=n_bins, weights=[weight], range=range2)

    return x_min,np.vstack((counts1,counts2)).reshape((-1,),order='F')

def weighted_mode(data:npt.ArrayLike,
                  bin_size:float,
                  weight:npt.ArrayLike=None,
                  scale:npt.ArrayLike=None) -> float:
    x_min,counts = weighted_histogram(data, bin_size, weight, scale)
    #print("counts1=",counts1)
    #print("counts2=",counts2)
    #print("counts=",counts)
    mode = np.argmax(counts)

    if mode == 0 or mode == counts.shape[0]-1:
        # no correction for peak at beginning and end of range
        return x_min + (mode + 0.5)*0.5*bin_size
    else:
        # Quadratic interpolation: f(x) = a*x^2 + b*x + c, f'(x) = 2*a*x + b,
        # so peak is at x = -b/(2*a)
        # y1 = f(-1) = a - b + c, y2 = f(0) = c, y3 = f(1) = a + b + c, so
        # y3 - y1 = 2*b, so b = (y3 - y1)/2,
        # y3 + y1 = 2*(a + c), so a = (y3 + y1)/2 - c = (y3 + y1)/2 - y2
        # So peak is at x = -((y3 - y1)/2)/(y3 + y1 - 2*y2)
        # For instance if y1 = y2, x = -((y3 - y1)/2)/(y3 - y1) = -1/2
        #              If y2 = y3, x = -((y3 - y1)/2)/(-y3 + y1) = 1/2
        # So -1/2 <= x at peak <= 1/2
        y1 = counts[mode-1]
        y2 = counts[mode]
        y3 = counts[mode+1]
        assert(y2 >= y1 and y2 >= y3)
        quadratic_correction = -0.5*(y3 - y1)/(y3 + y1 - 2.0*y2)
        #print("quadratic_correction=",quadratic_correction)
        return x_min + (mode + 0.5 + quadratic_correction)*0.5*bin_size
