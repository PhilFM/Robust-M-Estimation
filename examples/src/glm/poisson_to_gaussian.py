# Builds function to map a value sampled from a Poisson distribution to a Gaussian distribution
# This function is implemented as a bilinear interpolation on a 2D grid of values covering
# the values of the mean mu and the value x

import numpy as np
import scipy
import math

class Poisson_to_Gaussian:
    def __init__(
        self,
        max_mu: float,
        n_mu_values: int,
        max_x_mu_scale: float,
        n_x_values: int,
    ):
        self.__max_mu = max_mu
        self.__n_mu_values = n_mu_values
        self.__max_x_mu_scale = max_x_mu_scale
        self.__n_x_values = n_x_values

        self.__inv_array = None

    def _poisson(self,
                 x: float, mu: float) -> float:
        #print("x=",x,"mu=",mu)
        return math.pow(mu, x)*math.exp(-mu) / math.gamma(x+1.0)

    def _poisson_derivative(self,
                            x:float, mu: float) -> float:
        gam = math.gamma(x + 1.0)
        digam = scipy.special.digamma(x + 1.0)
        return math.pow(mu, x)*(math.log(mu) - digam)/gam 

    def _poisson_peak(self,
                      mu: float) -> float:
        bracket_l = mu - 1.0
        bracket_r = mu
        val_l = self._poisson_derivative(bracket_l, mu)
        val_r = self._poisson_derivative(bracket_r, mu)
        #if bracket_l == 0.0 and val_l <= 0.0:
        #    return 0.0

        #print("mu=",mu,"val_l=",val_l,"val_r=",val_r)
        assert(val_l > 0.0 and val_r < 0.0)
        for i in range(20):
            x_mid = 0.5*(bracket_l + bracket_r)
            val_mid = self._poisson_derivative(x_mid, mu)
            if val_mid > 0.0:
                bracket_l = x_mid
                val_l = val_mid
            else:
                bracket_r = x_mid
                val_r = val_mid

        return 0.5*(bracket_l + bracket_r)

    def build_inv_array(self) -> None:
        self.__x_step = np.zeros(self.__n_mu_values)
        self.__inv_array = np.zeros((self.__n_mu_values, self.__n_x_values))
        self.__mu_step = self.__max_mu/self.__n_mu_values
        for mui in range(self.__n_mu_values):
            # first find the maximum of the Poisson for this value of mu
            mu = (1+mui)*self.__mu_step
            mu_peak = self._poisson_peak(mu)
            mu_peak_poi = self._poisson(mu_peak, mu)

            self.__x_step[mui] = self.__max_x_mu_scale*mu/(self.__n_x_values - 1)
            for xi in range(self.__n_x_values):
                x = xi*self.__x_step[mui]
                poival = self._poisson(x, mu)

                #print("mu=",mu,"mu_peak=",mu_peak,"x=",x,"poival/mu_peak_poi=",poival/mu_peak_poi) #,"inv_array=",self.__inv_array[mui][xi])

                # invert Gaussian with zero mean and variance set to mu
                # exp(-y^2/mu) = poival/mu_peak_poi
                # y = sqrt(-mu*log(poival/mu_peak_poi))
                fid = -mu*math.log(poival/mu_peak_poi)
                assert(fid >= 0)
                self.__inv_array[mui][xi] = -math.sqrt(fid) if x < mu_peak else math.sqrt(fid)

        print("x_step=",self.__x_step)

    def _val(self,
             mu_idx: int,
             x: float) -> float:
        x_step = self.__x_step[mu_idx]
        xr = x/x_step
        x_idx = math.floor(xr)
        #print("x=",x,"xr=",xr,"x_idx=",x_idx,"x_alpha=",xr-x_idx)
        if x_idx < 0 or x_idx >= self.__n_x_values-1:
            return 0.0

        assert(x_idx >= 0 and x_idx < self.__n_x_values-1)
        x_alpha = xr - x_idx
        assert(x_alpha >= 0.0 and x_alpha <= 1.0)
        return x_alpha*self.__inv_array[mu_idx][x_idx+1] + (1.0-x_alpha)*self.__inv_array[mu_idx][x_idx]

    def gaussian_value(self,
                       mu: float,
                       x: float) -> float:
        mur = mu/self.__mu_step
        mu_idx = math.floor(mur)
        #print("mu=",mu,"mur=",mur,"mu_idx=",mu_idx,"mu_alpha=",mur-mu_idx)
        assert(mu_idx > 0 and mu_idx < self.__n_mu_values)
        mu_alpha = mur - mu_idx
        assert(mu_alpha >= 0.0 and mu_alpha <= 1.0)
        return mu_alpha*self._val(mu_idx, x) + (1.0 - mu_alpha)*self._val(mu_idx-1, x)
        
        
