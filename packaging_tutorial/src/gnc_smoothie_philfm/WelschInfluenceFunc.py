# Implementaton of Welsch IRLS influence function.
# From Holland & Welsch "Robust regression using iteratively reweighted least-squares"
# Communications in Statistics-theory and Methods, 6(9), 1977.
# We reverse the sign convention, to make the objective function a
# simple Gaussian rho(r) = exp(-r^2/(2*sigma*sigma))
# Wrap with the GNC_WelschParams class to use inside a GNC schedule, with the
# sigma parameter controlled by GNC.
# Can also be wrapped with NullParams and used with fixed sigma outside GNC.
import math


class WelschInfluenceFunc:
    def __init__(self, sigma: float = None):
        self.sigma = sigma

    # 1.0 means optimise for minimum, -1.0 means optimise for maximum
    def objective_func_sign(self) -> float:
        return -1.0

    # rho(r) = exp(-r^2/(2*sigma*sigma))
    def rho(self, rsqr: float, s: float) -> float:
        sigma = s * self.sigma
        return math.exp(-0.5 * (rsqr) / (sigma * sigma))

    # rho'(r)/r used in IRLS, as well as a and A terms in SupGaussNewton
    def rhop(self, rsqr: float, s: float) -> float:
        sigma = s * self.sigma
        invVar = 1.0 / (sigma * sigma)
        return -invVar * math.exp(-0.5 * rsqr * invVar)

    # (r*rho''(r) - rho'(r))/(r^3) used for B term in SupGaussNewton
    def Bterm(self, rsqr: float, s: float) -> float:
        sigma = s * self.sigma
        invVar = 1.0 / (sigma * sigma)
        return invVar * invVar * math.exp(-0.5 * rsqr * invVar)

    def summary(self) -> str:
        return "sigma=" + str(self.sigma)
