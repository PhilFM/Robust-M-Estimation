cimport cython
cimport numpy as np
np.import_array()

@cython.boundscheck(False)  # Deactivate bounds checking

def __weighted_mean(cython.double[:,:] data, cython.double[:] weight,
                    cython.double[:] scale, cython.double[:] X0):
    n: cython.Py_ssize_t = data.shape[0]
    sW: cython.double = 0.0
    X0[0] = X0[1] = 0.0
    i: cython.Py_ssize_t
    for i in range(n):
        w = weight[i] / (scale[i] * scale[i])
        X0[0] += w*data[i][0]
        X0[1] += w*data[i][1]
        sW += w

    X0[0] /= sW
    X0[1] /= sW

def line_fit_orthog_weighted_fit_sums(cython.double[:,:] data, cython.double[:] weight,
                                      cython.double[:] scale,
                                      cython.double[:] X0, cython.double[:,:] cov):
    __weighted_mean(data, weight, scale, X0)
    n: cython.Py_ssize_t = data.shape[0]
    i: cython.Py_ssize_t
    cov[0][0] = cov[0][1] = cov[1][1] = 0.0
    for i in range(n):
        dd: cython.double[2] = [data[i][0]-X0[0], data[i][1]-X0[1]]
        w = weight[i] / (scale[i] * scale[i])
        cov[0][0] += w*dd[0]*dd[0]
        cov[0][1] += w*dd[0]*dd[1]
        cov[1][1] += w*dd[1]*dd[1]

    cov[1][0] = cov[0][1]          
