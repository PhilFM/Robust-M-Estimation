cimport cython
from libc.math cimport sqrt
from libc.math cimport exp

# rho(r) = exp(-r^2/(2*sigma*sigma))
cdef inline cython.double welsch_rho(cython.double rsqr, cython.double inv_var):
    return exp(-0.5 * (rsqr) * inv_var)

# rho'(r)/r used in IRLS, as well as a and A terms in SupGaussNewton
cdef inline cython.double welsch_rhop(cython.double rsqr, cython.double inv_var):
    return -inv_var * exp(-0.5 * rsqr * inv_var)

# (r*rho''(r) - rho'(r))/(r^3) used for B term in SupGaussNewton
cdef inline cython.double welsch_Bterm(cython.double rsqr, cython.double inv_var):
    return inv_var * inv_var * exp(-0.5 * rsqr * inv_var)
