import numpy as np
import math
import sys
import scipy

sys.path.append("../misc")
from minimiser import minimiser

from line_fit_method import LineFitMethod
from line_fit_data_model import LineFitDataModel

class LineFitMethodMinDeriv(LineFitMethod):
    def __init__(self,
                 data_model:LineFitDataModel,
                 n_subdivisions:int = 21,
                 alpha_centre:float = 0.505,
                 beta_centre:float = 0.505,
                 gamma_centre:float = 3.0,
                 alpha_half_range:float = 0.49,
                 beta_half_range:float = 0.495,
                 gamma_half_range:float = 2.0,
                 n_iterations:int = 1,
                 n_alpha_divisions:int = 31,
                 n_beta_divisions:int = 31,
                 n_gamma_divisions:int = 31,
                 range_scale_factor:float = 7.0,
                 f_deriv_val_thres:float = 1.0e-5
                 ):
        LineFitMethod.__init__(
            self,
            data_model
        )
        self.__n_subdivisions = n_subdivisions
        self.__alpha_centre = alpha_centre
        self.__beta_centre  = beta_centre
        self.__gamma_centre = gamma_centre
        self.__alpha_half_range = alpha_half_range
        self.__beta_half_range  = beta_half_range
        self.__gamma_half_range = gamma_half_range
        self.__n_iterations = n_iterations
        self.__n_alpha_divisions = n_alpha_divisions
        self.__n_beta_divisions  = n_beta_divisions
        self.__n_gamma_divisions = n_gamma_divisions
        self.__range_scale_factor = range_scale_factor
        self.__f_deriv_val_thres = f_deriv_val_thres
        self.__small_beta_diff = 0.0001 # small value in beta direction for calculating derivatives
    
    def name(self):
        return "min_deriv"

    def F_derivs(self, x:np.ndarray, theta:float, delta:float):
        alpha = x[0]
        beta  = x[1]
        gamma = x[2]

        cost = math.cos(theta)
        sint = math.sin(theta)
        cosp = math.cos(theta+delta) # = cos(theta)*cos(delta)-sin(theta)*sin(delta)
        sinp = math.sin(theta+delta) # = sin(theta)*cos(delta)+cos(theta)*sin(delta)
    
        F_good_vals = [self._data_model.F_good(beta-2.0*self.__small_beta_diff, cost, sint, alpha, gamma, cosp, sinp),
                       self._data_model.F_good(beta-self.__small_beta_diff,     cost, sint, alpha, gamma, cosp, sinp),
                       self._data_model.F_good(beta,                     cost, sint, alpha, gamma, cosp, sinp),
                       self._data_model.F_good(beta+self.__small_beta_diff,     cost, sint, alpha, gamma, cosp, sinp),
                       self._data_model.F_good(beta+2.0*self.__small_beta_diff, cost, sint, alpha, gamma, cosp, sinp)]
        F_bad_vals = [self._data_model.F_bad(beta-2.0*self.__small_beta_diff, cost, sint, alpha, gamma, cosp, sinp),
                      self._data_model.F_bad(beta-self.__small_beta_diff,     cost, sint, alpha, gamma, cosp, sinp),
                      self._data_model.F_bad(beta,                     cost, sint, alpha, gamma, cosp, sinp),
                      self._data_model.F_bad(beta+self.__small_beta_diff,     cost, sint, alpha, gamma, cosp, sinp),
                      self._data_model.F_bad(beta+2.0*self.__small_beta_diff, cost, sint, alpha, gamma, cosp, sinp)]
        #print("F_good_vals=",F_good_vals)
        #print("F_bad_vals=",F_bad_vals)
        deriv_1_good = 0.5*(F_good_vals[3] - F_good_vals[1])/self.__small_beta_diff
        deriv_1_bad  = 0.5*(F_bad_vals[3]  - F_bad_vals[1] )/self.__small_beta_diff
        deriv_2_good = (F_good_vals[1] - 2.0*F_good_vals[2] + F_good_vals[3])/(self.__small_beta_diff*self.__small_beta_diff)
        deriv_2_bad  = (F_bad_vals[1]  - 2.0*F_bad_vals[2]  + F_bad_vals[3] )/(self.__small_beta_diff*self.__small_beta_diff)
        deriv_3_good = 0.5*(F_good_vals[4] - 2.0*F_good_vals[3] + 2.0*F_good_vals[1] - F_good_vals[0])/(self.__small_beta_diff*self.__small_beta_diff*self.__small_beta_diff)
        deriv_3_bad  = 0.5*(F_bad_vals[4]  - 2.0*F_bad_vals[3]  + 2.0*F_bad_vals[1]  - F_bad_vals[0] )/(self.__small_beta_diff*self.__small_beta_diff*self.__small_beta_diff)
        totF = F_good_vals[2] + F_bad_vals[2]
        derivs_good = (float(deriv_1_good/totF),float(deriv_2_good/totF),float(deriv_3_good/totF))
        derivs_bad  = (float(deriv_1_bad/totF), float(deriv_2_bad/totF), float(deriv_3_bad/totF) )
        #print("derivs_good:",derivs_good)
        #print("derivs_bad:",derivs_bad)
        return (derivs_good,derivs_bad)

    def F_deriv_sum(self, x:np.ndarray, theta:float, delta:float):
        # calculate basic squared derivative sum to minimise
        ((deriv_1_good,deriv_2_good,deriv_3_good),(deriv_1_bad,deriv_2_bad,deriv_3_bad)) = self.F_derivs(x, theta, delta)
        F_sum = (deriv_1_good + deriv_1_bad)**2 + (deriv_2_good + deriv_2_bad)**2 + (deriv_3_good + deriv_3_bad)**2

        # augment with terms to discourage stupid answers
        test_deriv_1 = deriv_1_good*deriv_1_bad
        if test_deriv_1 >= 0.0:
            F_sum += self._error_scale_sqr*test_deriv_1

        test_deriv_2 = abs(deriv_1_good)
        if test_deriv_2 < self._F_deriv_thres:
            F_sum += self._F_deriv_thres - test_deriv_2

        test_deriv_3 = abs(deriv_1_bad)
        if test_deriv_3 < self._F_deriv_thres:
            F_sum += self._F_deriv_thres - test_deriv_3

        test_deriv_4 = deriv_2_good*deriv_2_bad
        if test_deriv_4 > 0.0:
            F_sum += self._error_scale_sqr*test_deriv_4

        test_deriv_5 = deriv_3_good*deriv_3_bad
        if test_deriv_5 > 0.0:
            F_sum += self._error_scale_sqr*test_deriv_5

        return F_sum

    def apply(self, theta:float, delta:float):
        initial_centre = [self.__alpha_centre, self.__beta_centre, self.__gamma_centre]
        initial_half_range = [self.__alpha_half_range, self.__beta_half_range, self.__gamma_half_range]

        best_sample,best_val = minimiser(self.F_deriv_sum, initial_centre, initial_half_range,
                                         [self.__n_subdivisions,self.__n_subdivisions,self.__n_subdivisions], # n_samples
                                         args=(theta, delta),
                                         n_iterations=self.__n_iterations,
                                         initial_n_samples=[self.__n_alpha_divisions,self.__n_beta_divisions,self.__n_gamma_divisions],
                                         scale_factor=self.__range_scale_factor) #, debug=True)
        #print("  delta=",delta,"best_val=",best_val,"best_sample=",best_sample,"Fval=",F_tot(best_sample[1], F_fac, data_model, theta, best_sample[0], best_sample[2], delta))
        bounds = []
        for i in range(3):
            bounds.append((initial_centre[i]-initial_half_range[i],initial_centre[i]+initial_half_range[i]))

        init_val = self.F_deriv_sum(best_sample, theta, delta)
        sol = scipy.optimize.minimize(self.F_deriv_sum, best_sample, method="Nelder-Mead", tol=1.e-10, bounds=bounds, args=(theta, delta))
        print("  delta=",delta,"best_val=",sol.fun,"init_val=",init_val, "final   sol=",sol.x,"init sol=",best_sample,"Fval=",self._data_model.F_tot(sol.x[1], theta, sol.x[0], sol.x[2], delta),"success=",sol.success)
        if sol.fun <= init_val and sol.fun < self.__f_deriv_val_thres: # and sol.success
            return sol.x, sol.fun
        else:
            return None,None
