# Quadratic influence function rho(r) = r*r/2
# This implements standard least-squares so not really relevant to IRLS
# (because not robust) but useful to have as a reference.
class QuadraticInfluenceFunc:
    def __init__(self):
        pass

    # 1.0 means optimise for minimum, -1.0 means optimise for maximum
    def objective_func_sign(self) -> float:
        return 1.0

    # rho(r)
    def rho(self, rsqr: float, s: float) -> float:
        return 0.5 * rsqr / (s * s)

    # rho'(r)/r used in IRLS, as well as a and A terms in SupGaussNewton
    def rhop(self, rsqr: float, s: float) -> float:
        return 1.0 / (s * s)

    # (r*rho''(r) - rho'(r))/(r^3) used for B term in SupGaussNewton
    def Bterm(self, rsqr: float, s: float) -> float:
        return 0.0

    def summary(self) -> str:
        return ""
