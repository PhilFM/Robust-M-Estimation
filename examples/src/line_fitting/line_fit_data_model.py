import math

class LineFitDataModel:
    def __init__(self, sup_gn_q):
        self._sup_gn_q = sup_gn_q
        self._F_fac = 1.0/math.sqrt(1.0 + sup_gn_q**2)

    def sup_gn_q(self):
        return self._sup_gn_q

    def F_fac(self):
        return self._F_fac

    def F_tot(self, beta:float, theta:float, alpha:float, gamma:float, delta:float):
        cost = math.cos(theta)
        sint = math.sin(theta)
        cosp = math.cos(theta+delta) # = cos(theta)*cos(delta)-sin(theta)*sin(delta)
        sinp = math.sin(theta+delta) # = sin(theta)*cos(delta)+cos(theta)*sin(delta)
    
        return self.F_good(beta, cost, sint, alpha, gamma, cosp, sinp) + self.F_bad(beta, cost, sint, alpha, gamma, cosp, sinp)

