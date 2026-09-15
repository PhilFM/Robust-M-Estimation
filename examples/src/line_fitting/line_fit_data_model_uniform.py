import math

from line_fit_data_model import LineFitDataModel

# This class provides supports line fitting assuming a uniform distrubtion of data x values over a range
class LineFitDataModelUniform(LineFitDataModel):
    def __init__(self,
                 sup_gn_q:float):
        LineFitDataModel.__init__(
            self,
            sup_gn_q
        )

        self.__F_canon_thres = 0.00001
        self.__sqrt_pi = math.sqrt(math.pi)
        self.__sqrt_2 = math.sqrt(2.0)
        self.__inv_sqrt_2 = 1.0/math.sqrt(2.0)

    def name(self):
        return "uniform"

    def F_canon(self, u:float, v:float):
        upv = u+v
        if abs(upv) < self.__F_canon_thres:
            return (2.0 + 0.66666667*(u*v-u*u-v*v))/self.__sqrt_pi
        else:
            return (math.erf(u) + math.erf(v))/upv

    # In general, F_g_uniform(a,b) = (1-alpha)*Fc(ug,bg)*sqrt(2)*F_fac
    # where ug = -(a*D+b)/(sqrt(2)*sigma)
    #       vg = (a*xbl+b)/(sqrt(2)*sigma)
    #       xbl = -D + 2*alpha*D
    #       F_fac = 1/sqrt(1 + q*q)
    # But in canonical terms, D=sigma = 1 and this simplifies to
    #       ug = -(a+b)/sqrt(2)
    #       vg =  (a*(-1+2*alpha)+b)/sqrt(2)
    # We also have
    #       phi = theta+delta
    #       cp = cos(phi) = ct*cd-st*sd
    #       sp = sin(phi) = st*cd+ct*sd
    #       a = beta*gamma*cp
    #       b = beta*gamma*sp
    # So we have
    #       ug = -beta*gamma*(cp+sp)/sqrt(2)
    #       vg = beta*gamma*(cp*(-1+2*alpha)+sp)/sqrt(2)
    def F_good(self, beta:float, cost:float, sint:float, alpha:float, gamma:float, cosp:float, sinp:float):
        ug = -self.__inv_sqrt_2*beta*gamma*(cosp + sinp)*self._F_fac
        vg =  self.__inv_sqrt_2*beta*gamma*((-1.0 + 2.0*alpha)*cosp + sinp)*self._F_fac
        return (1.0-alpha)*self.F_canon(ug, vg)*self._F_fac*self.__sqrt_2

    # In general, F_b_uniform(a,b) = alpha*Fc(ub,vb)*sqrt(2)
    # where ub = -((a-ab)*D + bb-b)/(sqrt(2)*sigma)
    #       vb =  ((ab-a)*xbl + bb-b)/(sqrt(2)*sigma)
    # In canonical terms
    # where ub =  (ab-a - bb+b)/sqrt(2)
    #       vb =  ((ab-a)*(-1+2*alpha) + bb-b)/sqrt(2)
    # Also we have
    #       ab = gamma*ct
    #       bb = gamma*st
    #       a = beta*gamma*cp
    #       b = beta*gamma*sp
    # So we have
    #       ub = gamma*(beta*(sp-cp) + ct - st)/sqrt(2)
    #       vb = gamma*((ct-beta*cp)*(-1+2*alpha) + st-beta*sp)/sqrt(2)
    def F_bad(self, beta: float, cost:float, sint:float, alpha:float, gamma:float, cosp:float, sinp:float):
        ub = self.__inv_sqrt_2*gamma*(beta*(sinp-cosp) + cost - sint)
        vb = self.__inv_sqrt_2*gamma*((cost-beta*cosp)*(-1.0 + 2.0*alpha) + sint-beta*sinp)
        return alpha*self.F_canon(ub, vb)*self.__sqrt_2

