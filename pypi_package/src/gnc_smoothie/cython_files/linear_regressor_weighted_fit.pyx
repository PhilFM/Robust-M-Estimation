cimport cython

cimport numpy as np

np.import_array()

@cython.boundscheck(False)  # Deactivate bounds checking

cdef inline increment_weighted_fit_sums(cython.double[:,:] data_item, cython.double w,
                                        cython.double[:,:] atot, cython.double[:,:,:] Atot):
    j: cython.Py_ssize_t
    k: cython.Py_ssize_t
    l: cython.Py_ssize_t
    dim: cython.Py_ssize_t = data_item.shape[1]-1
    for j in range(data_item.shape[0]):
        for k in range(dim):
            atot[j][k] += w*data_item[j][k]*data_item[j][dim]
            for l in range(k,dim):
                Atot[j][k][l] += w*data_item[j][k]*data_item[j][l]

            Atot[j][k][dim] += w*data_item[j][k]    

        atot[j][dim] += w*data_item[j][dim]
        Atot[j][dim][dim] += w

def linear_regressor_weighted_fit(cython.double[:,:,:] data, cython.double[:] weight,
                                  cython.double[:] scale,
                                  cython.double[:,:] atot, cython.double[:,:,:] Atot):
    i: cython.Py_ssize_t
    for i in range(data.shape[0]):
        increment_weighted_fit_sums(data[i], weight[i] / (scale[i]*scale[i]), atot, Atot)

    j: cython.Py_ssize_t
    k: cython.Py_ssize_t
    l: cython.Py_ssize_t
    for j in range(data.shape[1]):
        for k in range(data.shape[2]):
            for l in range(data.shape[2]):
                Atot[j][l][k] = Atot[j][k][l]
            
