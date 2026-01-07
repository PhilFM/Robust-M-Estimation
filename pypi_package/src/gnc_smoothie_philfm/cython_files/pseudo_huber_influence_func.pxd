cimport cython
from libc.math cimport sqrt
from libc.math cimport pow

# rho(r) = sqrt(1 + r^2/(sigma*sigma)) - 1.0
cdef inline cython.double pseudo_huber_rho(cython.double rsqr, cython.double inv_var):
    return sqrt(1.0 + rsqr * inv_var) - 1.0

# rho'(r)/r used in IRLS, as well as a and A terms in SupGaussNewton
cdef inline cython.double pseudo_huber_rhop(cython.double rsqr, cython.double inv_var):
    return inv_var / sqrt(1.0 + rsqr * inv_var)

# (r*rho''(r) - rho'(r))/(r^3) used for B term in SupGaussNewton
cdef inline cython.double pseudo_huber_Bterm(cython.double rsqr, cython.double inv_var):
    return -inv_var * inv_var * pow(1.0 + rsqr * inv_var, -1.5)
