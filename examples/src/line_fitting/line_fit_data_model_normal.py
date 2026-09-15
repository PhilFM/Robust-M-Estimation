import math
import scipy

from line_fit_data_model import LineFitDataModel

# This class provides supports line fitting assuming a Gaussian distrubtion of data x values
class LineFitDataModelNormal(LineFitDataModel):
    def __init__(self,
                 sup_gn_q:float):
        LineFitDataModel.__init__(
            self,
            sup_gn_q
        )
        self.__sqrt_2 = math.sqrt(2.0)

    def name(self):
        return "normal"

    def F_good(self, beta:float, cost:float, sint:float, alpha:float, gamma:float, cosp:float, sinp:float):
        x0 = scipy.special.erfinv(2.0*alpha-1.0) # so alpha = 0.5*(1+erf(x0/sigma_l))
        (a,b) = (beta*gamma*cosp, beta*gamma*sinp)

        u = math.sqrt(self._F_fac*self._F_fac*a*a + 1.0)/self.__sqrt_2
        v = 0.5*self._F_fac*self._F_fac*a*b/u
        w = 0.5*self._F_fac*self._F_fac*b*b - v*v
        return self._F_fac*math.exp(-w)*(1.0 - math.erf(u*x0+v))/u

    def F_bad(self, beta: float, cost:float, sint:float, alpha:float, gamma:float, cosp:float, sinp:float):
        x0 = scipy.special.erfinv(2.0*alpha-1.0) # so alpha = 0.5*(1+erf(x0/sigma_l))
        (ab,bb) = (gamma*cost, gamma*sint)
        (a,b) = (beta*gamma*cosp, beta*gamma*sinp)

        ad = ab-a
        bd = bb-b
        u = math.sqrt(ad*ad + 1.0)/self.__sqrt_2
        v = 0.5*ad*bd/u
        w = 0.5*bd*bd - v*v
        return math.exp(-w)*(1.0 + math.erf(u*x0+v))/u

