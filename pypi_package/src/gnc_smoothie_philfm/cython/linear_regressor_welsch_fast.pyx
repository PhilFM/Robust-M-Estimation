cimport cython

from cython.cimports.src.gnc_smoothie_philfm.cython.welsch_influence_func import welsch_rho, welsch_rhop, welsch_Bterm
from cython.cimports.src.gnc_smoothie_philfm.cython.linear_regressor_fast import linear_regressor_calc_residual, linear_regressor_calc_grad, increment_weighted_deriv_sums

@cython.boundscheck(False)  # Deactivate bounds checking

def linear_regressor_welsch_objective_func(cython.double sigma,
                                           cython.double[:] model,
                                           cython.double[:,:,:] data, cython.double[:] weight,
                                           cython.double[:] scale,
                                           cython.double[:] residual) -> cython.double:
    tot: cython.double = 0.0
    i: cython.Py_ssize_t
    for i in range(data.shape[0]):
        rsqr: cython.double = linear_regressor_calc_residual(model, data[i], residual)
        this_sigma: cython.double = scale[i] * sigma
        inv_var: cython.double = 1.0 / (this_sigma * this_sigma)
        tot += weight[i] * welsch_rho(rsqr, inv_var)

    return tot

def linear_regressor_welsch_weighted_derivs(cython.double sigma, cython.double lambda_val,
                                            cython.double[:] model,
                                            cython.double[:,:,:] data, cython.double[:] weight,
                                            cython.double[:] scale,
                                            cython.double[:] residual, cython.double[:,:] grad,
                                            cython.double[:] atot, cython.double[:,:] AlBtot):
    i: cython.Py_ssize_t
    for i in range(data.shape[0]):
        rsqr: cython.double = linear_regressor_calc_residual(model, data[i], residual)
        linear_regressor_calc_grad(data[i], residual, grad)
        this_sigma: cython.double = scale[i] * sigma
        inv_var: cython.double = 1.0 / (this_sigma * this_sigma)
        rhop: cython.double = welsch_rhop(rsqr, inv_var)
        Bterm: cython.double = welsch_Bterm(rsqr, inv_var)
        increment_weighted_deriv_sums(lambda_val, data[i], grad, rhop, Bterm, weight[i], atot, AlBtot)

    # fill in lower diagonal entries in AlBtot
    k: cython.Py_ssize_t
    l: cython.Py_ssize_t
    for k in range(data.shape[1]*data.shape[2]):
        for l in range(k+1,data.shape[1]*data.shape[2]):
            AlBtot[l][k] = AlBtot[k][l]

def linear_regressor_welsch_update_weights(cython.double sigma,
                                           cython.double[:] model,
                                           cython.double[:,:,:] data, cython.double[:] weight,
                                           cython.double[:] scale,
                                           cython.double[:] residual, cython.double[:] new_weight):
    i: cython.Py_ssize_t
    for i in range(data.shape[0]):
        rsqr: cython.double = linear_regressor_calc_residual(model, data[i], residual)
        this_sigma: cython.double = scale[i] * sigma
        inv_var: cython.double = 1.0 / (this_sigma * this_sigma)
        new_weight[i] = -weight[i]*welsch_rhop(rsqr, inv_var)
