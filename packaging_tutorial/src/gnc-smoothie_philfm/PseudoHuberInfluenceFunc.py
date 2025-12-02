# "Pseudo-Huber" IRLS influence function, introduced by Charbonnier et al. in
# "Deterministic edge-preserving regularization in computed imaging", PAMI 6(2), 1997.
# It approximates the original Huber IRLS influence function, "Robust Estimation of a
# Location Parameter", The Annals of Mathematical Statistics 35(1), 1964.
# The pseudo-Huber influence function is differentiable everywhere and easier
# to work with mathematically.
import math


class PseudoHuberInfluenceFunc:
    def __init__(self, sigma: float = None):
        self.sigma = sigma

    # 1.0 means optimise for minimum, -1.0 means optimise for maximum
    def objective_func_sign(self) -> float:
        return 1.0

    # rho(r) = sqrt(1 + r^2/(sigma*sigma)) - 1.0
    def rho(self, rsqr: float, s: float) -> float:
        sigma = s * self.sigma
        invVar = 1.0 / (sigma * sigma)
        return math.sqrt(1.0 + rsqr * invVar) - 1.0

    # rho'(r)/r used in IRLS, as well as a and A terms in SupGaussNewton
    def rhop(self, rsqr: float, s: float) -> float:
        sigma = s * self.sigma
        invVar = 1.0 / (sigma * sigma)
        return invVar / math.sqrt(1.0 + rsqr * invVar)

    # (r*rho''(r) - rho'(r))/(r^3) used for B term in SupGaussNewton
    def Bterm(self, rsqr: float, s: float) -> float:
        sigma = s * self.sigma
        invVar = 1.0 / (sigma * sigma)
        return -invVar * invVar * math.pow(1.0 + rsqr * invVar, -1.5)

    def summary(self) -> str:
        return "sigma=" + str(self.sigma)
