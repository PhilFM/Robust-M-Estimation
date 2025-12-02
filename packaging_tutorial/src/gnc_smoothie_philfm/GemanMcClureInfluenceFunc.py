# Geman-McClure IRLS influence function introduced in Geman & Geman "Bayesian image analysis",
# in "Disordered systems and biological organization", 1986.
import math


class GemanMcClureInfluenceFunc:
    def __init__(self, sigma: float = None):
        self.sigma = sigma

    # 1.0 means optimise for minimum, -1.0 means optimise for maximum
    def objective_func_sign(self) -> float:
        return 1.0

    #               r^2
    # rho(r) = -------------
    #          sigma^2 + r^2
    #
    #  where r is the magnitude of the residual vector, r = ||rvec||
    # rho(r)
    def rho(self, rsqr: float, s: float) -> float:
        sigma = s * self.sigma
        var = sigma * sigma
        return rsqr / (rsqr + var)

    # rho'(r)/r used in IRLS, as well as a and A terms in SupGaussNewton
    def rhop(self, rsqr: float, s: float) -> float:
        sigma = s * self.sigma
        var = sigma * sigma
        return 2.0 * var / ((rsqr + var) ** 2.0)

    # (r*rho''(r) - rho'(r))/(r^3) used for B term in SupGaussNewton
    def Bterm(self, rsqr: float, s: float) -> float:
        sigma = s * self.sigma
        var = sigma * sigma
        return -8.0 * var / math.pow(rsqr + var, 3.0)

    def summary(self) -> str:
        return "sigma=" + str(self.sigma)
