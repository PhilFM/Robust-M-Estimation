cimport cython
from libc.math cimport pow
from libc.math cimport log
from libc.math cimport sqrt

# we have d(rho)/d(r)
#         ----------- = (max(sr,epsilon))^(p-2) where sr = rscale*r
#              r
# So
#    d(rho)
#    ------ = r*(max(sr,epsilon)^(p-2))
#     d(r)
#           = r*epsilon^(p-2)                if rscale*r < epsilon
#             rscale^(p-2)*r^(p-1)           otherwise
#
# So rho(x) = epsilon^(p-2)*r^2/2              if rscale*r < epsilon
#             K + rscale^(p-2)*log(r)          if rscale*r >= epsilon and p=0
#             K + sr^p/(rscale*rscale*p)       otherwise
#
# where K = epsilon^(p-2)*(epsilon/rscale)^2/2 - rscale^(p-2)*log(epsilon/rscale)            if p = 0
#       K = epsilon^(p-2)*(epsilon/rscale)^2/2 - epsilon^p/(rscale*rscale*p)                 if p != 0
# or
#
# where K = epsilon^p*rscale^(-2)/2 - rscale^(p-2)*log(epsilon/rscale)    if p = 0
#       K = epsilon^p*(1/2 - 1/p)/(rscale*rscale)                         if p != 0
cdef inline cython.double gnc_irls_p_rho(cython.double rsqr, cython.double p, cython.double rscale, cython.double epsilon):
    srsqr = rsqr * rscale * rscale
    sr = sqrt(srsqr)
    if sr < epsilon:
        return 0.5 * pow(epsilon, p - 2.0) * rsqr
    elif p == 0.0:  # and sr >= epsilon
        K = 0.5 * pow(epsilon, p) * pow(rscale, -2.0) - pow(rscale, p - 2.0) * log(epsilon / rscale)
        return K + pow(rscale, p - 2.0) * log(sr / rscale)
    else:  # sr >= epsilon and p != 0
        K = pow(epsilon, p) * (0.5 - 1.0 / p) / (rscale * rscale)
        return K + pow(sr, p) / (rscale * rscale * p)

# rho'(r)/r used in IRLS, as well as a and A terms in SupGaussNewton
cdef inline cython.double gnc_irls_p_rhop(cython.double rsqr, cython.double p, cython.double rscale, cython.double epsilon):
    srsqr = rsqr * rscale * rscale
    sr = sqrt(srsqr)
    return pow(max(sr, epsilon), p - 2.0)

# (r*rho''(r) - rho'(r))/(r^3) used for B term in SupGaussNewton
cdef inline cython.double gnc_irls_p_Bterm(cython.double rsqr, cython.double p, cython.double rscale, cython.double epsilon):
    srsqr = rsqr * rscale * rscale
    sr = sqrt(srsqr)
    if sr < epsilon:
        return 0.0
    else:
        return (p - 2.0) * pow(sr, p - 2.0) / rsqr

    
