cimport cython

from cython.cimports.src.gnc_smoothie_philfm.cython_files.gnc_irls_p_influence_func import gnc_irls_p_rho, gnc_irls_p_rhop, gnc_irls_p_Bterm
from cython.cimports.src.gnc_smoothie_philfm.cython_files.linear_regressor_fast import linear_regressor_calc_residual, linear_regressor_calc_grad, increment_weighted_deriv_sums

@cython.boundscheck(False)  # Deactivate bounds checking

def linear_regressor_gnc_irls_p_objective_func(cython.double p, cython.double rscale, cython.double epsilon,
                                               cython.double[:] model,
                                               cython.double[:,:,:] data, cython.double[:] weight,
                                               cython.double[:] scale,
                                               cython.double[:] residual) -> cython.double:
    tot: cython.double = 0.0
    i: cython.Py_ssize_t
    for i in range(data.shape[0]):
        rsqr: cython.double = linear_regressor_calc_residual(model, data[i], residual)
        tot += weight[i] * gnc_irls_p_rho(rsqr, p, rscale, scale[i]*epsilon)

    return tot

def linear_regressor_gnc_irls_p_weighted_derivs(cython.double p, cython.double rscale, cython.double epsilon,
                                                cython.double lambda_b, cython.double[:] model,
                                                cython.double[:,:,:] data, cython.double[:] weight,
                                                cython.double[:] scale,
                                                cython.double[:] residual, cython.double[:,:] grad,
                                                cython.double[:] atot, cython.double[:,:] AlBtot):
    i: cython.Py_ssize_t
    for i in range(data.shape[0]):
        rsqr: cython.double = linear_regressor_calc_residual(model, data[i], residual)
        linear_regressor_calc_grad(data[i], residual, grad)
        rhop: cython.double = gnc_irls_p_rhop(rsqr, p, rscale, scale[i]*epsilon)
        Bterm: cython.double = gnc_irls_p_Bterm(rsqr, p, rscale, scale[i]*epsilon)
        increment_weighted_deriv_sums(lambda_b, data[i], grad, rhop, Bterm, weight[i], atot, AlBtot)

    # fill in lower diagonal entries in AlBtot
    k: cython.Py_ssize_t
    l: cython.Py_ssize_t
    for k in range(data.shape[1]*data.shape[2]):
        for l in range(k+1,data.shape[1]*data.shape[2]):
            AlBtot[l][k] = AlBtot[k][l]

def linear_regressor_gnc_irls_p_update_weights(cython.double p, cython.double rscale, cython.double epsilon,
                                               cython.double[:] model,
                                               cython.double[:,:,:] data, cython.double[:] weight,
                                               cython.double[:] scale,
                                               cython.double[:] residual, cython.double[:] new_weight):
    i: cython.Py_ssize_t
    for i in range(data.shape[0]):
        rsqr: cython.double = linear_regressor_calc_residual(model, data[i], residual)
        new_weight[i] = weight[i]*gnc_irls_p_rhop(rsqr, p, rscale, scale[i]*epsilon)
