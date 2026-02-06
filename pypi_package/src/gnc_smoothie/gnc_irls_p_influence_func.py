# Implementation of IRLS-p IRLS influence function and GNC schedule.
# From Peng et al. "On the Convergence of IRLS and Its Variants in Outlier-Robust Estimation", CVPR 2023.
# Additional feature is the "rscale" parameter used to rescale data.
import math


class GNC_IRLSpInfluenceFunc:
    def __init__(self, p: float = None, rscale: float = None, epsilon: float = None):
        self.p = p
        self.rscale = rscale
        self.epsilon = epsilon

    # 1.0 means optimise for minimum, -1.0 means optimise for maximum
    def objective_func_sign(self) -> float:
        return 1.0

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
    def rho(self, rsqr: float, s: float) -> float:
        p = self.p
        rscale = self.rscale
        epsilon = s * self.epsilon
        srsqr = rsqr * rscale * rscale
        sr = math.sqrt(srsqr)
        if sr < epsilon:
            return 0.5 * math.pow(epsilon, p - 2.0) * rsqr
        elif p == 0.0:  # and sr >= epsilon
            K = 0.5 * math.pow(epsilon, p) * math.pow(rscale, -2.0) - math.pow(
                rscale, p - 2.0
            ) * math.log(epsilon / rscale)
            return K + math.pow(rscale, p - 2.0) * math.log(sr / rscale)
        else:  # sr >= epsilon and p != 0
            K = math.pow(epsilon, p) * (0.5 - 1.0 / p) / (rscale * rscale)
            return K + math.pow(sr, p) / (rscale * rscale * p)

    # rho'(r)/r used in IRLS, as well as a and A terms in SupGaussNewton
    def rhop(self, rsqr: float, s: float) -> float:
        p = self.p
        epsilon = s * self.epsilon
        rscale = self.rscale
        srsqr = rsqr * rscale * rscale
        sr = math.sqrt(srsqr)
        return math.pow(max(sr, epsilon), p - 2.0)

    # (r*rho''(r) - rho'(r))/(r^3) used for B term in SupGaussNewton
    def Bterm(self, rsqr: float, s: float):
        p = self.p
        epsilon = s * self.epsilon
        rscale = self.rscale
        srsqr = rsqr * rscale * rscale
        sr = math.sqrt(srsqr)
        if sr < epsilon:
            return 0.0
        else:
            return (p - 2.0) * math.pow(sr, p - 2.0) / rsqr

    def summary(self) -> str:
        return (
            "p="
            + str(self.p)
            + " rscale="
            + str(self.rscale)
            + " epsilon="
            + str(self.epsilon)
        )
