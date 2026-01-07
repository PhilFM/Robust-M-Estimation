cimport cython

cimport numpy as np

np.import_array()

from cython.cimports.src.cython_files.welsch_influence_func import welsch_rho, welsch_rhop, welsch_Bterm

@cython.boundscheck(False)  # Deactivate bounds checking

def line_fit_orthog_welsch_objective_func(cython.double sigma, cython.double a, cython.double b, cython.double c,
                                          cython.double[:,:] data, cython.double[:] weight,
                                          cython.double[:] scale) -> cython.double:
    tot: cython.double = 0.0
    n: cython.Py_ssize_t = data.shape[0]
    i: cython.Py_ssize_t
    for i in range(n):
        x: cython.double = data[i][0]
        y: cython.double = data[i][1]
        residual: cython.double = a*x + b*y + c
        this_sigma: cython.double = scale[i] * sigma
        inv_var: cython.double = 1.0 / (this_sigma * this_sigma)
        tot += weight[i] * welsch_rho(residual*residual, inv_var)

    return tot

def line_fit_orthog_welsch_update_weights(cython.double sigma,
                                          cython.double a, cython.double b, cython.double c,
                                          cython.double[:,:] data, cython.double[:] weight,
                                          cython.double[:] scale, cython.double[:] new_weight):
    n: cython.Py_ssize_t = data.shape[0]
    i: cython.Py_ssize_t
    for i in range(n):
        x: cython.double = data[i][0]
        y: cython.double = data[i][1]
        residual: cython.double = a*x + b*y + c
        rsqr: cython.double = residual*residual
        this_sigma: cython.double = scale[i] * sigma
        inv_var: cython.double = 1.0 / (this_sigma * this_sigma)
        new_weight[i] = -weight[i]*welsch_rhop(rsqr, inv_var)
