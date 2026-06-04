# mu is the Poisson mean/variance
# GLM link function is log(), so a.x + b = ln(mu)
# So exp(a.x + b) = mu
#
# Poisson distribution is P(k) = lambda^k e^-lambda / k!
#                              = lambda^k e^-lambda / gamma(k+1)
# where lambda is the mean
#
#                                          gamma(r+k)  (  r  )^r (  m  )^k
# negative binomial distribution is P(k) = ----------- (-----)   (-----)
#                                          k! gamma(r) (r + m)   (r + m)
#
#                         r(1-p)
# where m is the mean m = ------
#                            p
#
# r is the number of successes, n-r is the number of failures
#
# We have then p = r/(m+r), 1-p = m/(m+r)
# Variance is then m + m^2/r, or alpha = 1/r and variance = m + alpha*m^2
#
# gamma(n) = (n-1)!
#
# Converges on Poisson as p --> 1
#
# Alternative definition in terms of mean mu and variance sigma^2:
#
# Pr(X == k) = (k + mu^2/(sigma^2-mu) - 1) (1 - mu/sigma^2)^k (mu/sigma^2)^(mu^2/(sigma^2-mu))
#              (          k              )
#
# p = mu/sigma^2, r = mu^2/(sigma^2-mu)
# And p is in the range [0,1] so sigma >= sqrt(mu)
#
#              (k + mu^2/(sigma^2-mu) - 1)!
# Pr(X == k) = ---------------------------- (1 - mu/sigma^2)^k (mu/sigma^2)^(mu^2/(sigma^2-mu))
#               k! (mu^2/(sigma^2-mu) - 1)!
#
#                  gamma(k + mu^2/(sigma^2-mu))
#            = ------------------------------------ (1 - mu/sigma^2)^k (mu/sigma^2)^(mu^2/(sigma^2-mu))
#               gamma(k+1) gamma(mu^2/(sigma^2-mu))

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import math

if __name__ == "__main__":
    import sys
    sys.path.append("../../../pypi_package/src")

from gnc_smoothie.sup_gauss_newton import SupGaussNewton
from gnc_smoothie.gnc_null_params import GNC_NullParams

# Welsch
from gnc_smoothie.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie.welsch_influence_func import WelschInfluenceFunc

from poisson_to_gaussian import Poisson_to_Gaussian

def _poisson(x: float, mu: float) -> float:
    #return scipy.stats.poisson.pmf(x, mu)
    return math.pow(mu, x)*math.exp(-mu) / math.gamma(x+1.0)

def _neg_log_poisson(x: float, mu: float) -> float:
    # log(poisson func) = log(gamma(x+1)) - log(mu^x) + mu
    #                   = log(gamma(x+1)) - x*log(mu) + mu
    return scipy.special.loggamma(x+1.0) - x*math.log(mu) + mu

def _negative_binomial(x: float, mu: float, sigma: float) -> float:
    mu_sqr = mu ** 2.0
    var = sigma ** 2.0
    diff = var - mu
    #print("diff=",diff)
    #print(x + mu_sqr/diff)
    binom_val = scipy.special.binom(x + mu_sqr/diff - 1.0, x)
    #fid1 = math.gamma(x + mu_sqr/diff)
    #fid2 = math.gamma(x + 1.0)
    #fid3 = math.gamma(mu_sqr/diff)
    #print("test binom:",binom_val,"test2=",fid1/(fid2*fid3))
    return binom_val * math.pow(1.0 - mu/var, x) * math.pow(mu/var, mu_sqr/diff)

def _neg_log_negative_binomial(x: float, mu: float, sigma: float) -> float:
    mu_sqr = mu ** 2.0
    var = sigma ** 2.0
    diff = var - mu
    binom_val = scipy.special.binom(x + mu_sqr/diff - 1.0, x)
    binom_log = math.log(binom_val)
    fid_log = math.log(1.0 - mu/var)
    fid2_log = math.log(mu/var)
    return -binom_log - x*fid_log - mu_sqr*fid2_log/diff
    #return -math.log(binom_val * math.pow(1.0 - mu/var, x) * math.pow(mu/var, mu_sqr/diff))

def _show_poisson():
    mu_pop = 7.0 # population distribution mean
    x_min = 0.0
    x_max = 5.0*mu_pop
    x_list = np.linspace(x_min, x_max, num=300)
    hmfv = np.vectorize(_poisson, excluded={"mu"})
    plt.plot(x_list, hmfv(x_list, mu=mu_pop))
    plt.show()
           
def _show_neg_log_poisson():
    mu_pop = 0.3 # population distribution mean
    x_min = 0.0
    x_max = 50000.0*mu_pop
    x_list = np.linspace(x_min, x_max, num=300)
    hmfv = np.vectorize(_neg_log_poisson, excluded={"mu"})
    plt.plot(x_list, hmfv(x_list, mu=mu_pop))
    plt.show()
           
def _show_negative_binomial():
    x_min = 0.0
    x_max = 10.0
    x_list = np.linspace(x_min, x_max, num=300)
    mu_pop = 5.0 # population distribution mean
    sigma_pop = 3.0 # population distribution standard deviation
    assert(sigma_pop ** 2.0 >= mu_pop)
    hmfv = np.vectorize(_negative_binomial, excluded={"mu","sigma"})
    plt.plot(x_list, hmfv(x_list, mu=mu_pop, sigma=sigma_pop))
    plt.show()
           
def _show_neg_log_negative_binomial():
    x_min = 0.0
    x_max = 10.0
    x_list = np.linspace(x_min, x_max, num=300)
    mu_pop = 4.0 # population distribution mean
    sigma_pop = 4.0 # population distribution standard deviation
    assert(sigma_pop ** 2.0 >= mu_pop)
    hmfv = np.vectorize(_neg_log_negative_binomial, excluded={"mu","sigma"})
    #ax = plt.gca()
    #plt.box(False)
    #ax.set_ylim((0.0, y_max))
    plt.plot(x_list, hmfv(x_list, mu=mu_pop, sigma=sigma_pop))
    plt.show()
           
class OverdispersedPoissonInfluenceFunc:
    def __init__(self, mu: float = None, sigma: float = None):
        self.__sigma = sigma

    def rho(self, rsqr: float, s: float) -> float:
        sigma = s * self.__sigma
        return math.exp(-0.5 * (rsqr) / (sigma * sigma))

    def summary(self) -> str:
        return "sigma=" + str(self.__sigma)

class PoissonFitter:
    def __init__(
        self,
        mu: float,
    ):
        self.__mu = mu
        return n_calc

    def cache_model(self, model, model_ref=None):
        self.__model = np.copy(model)

    def residual(self, data_item) -> np.array:
        assert(data_item.shape[0] == self.__model.shape[0]-1)
        return np.dot(self.__model[0:-1], data_item) + self.__model[-1] - self.__mu

def _log_gamma(x: float) -> float:
    return math.log(math.gamma(x))

def _show_log_gamma():
    x_min = 1.0
    x_max = 100.0
    x_list = np.linspace(x_min, x_max, num=300)
    hmfv = np.vectorize(_log_gamma)
    plt.plot(x_list, hmfv(x_list))
    plt.show()
    

def _fit_poisson(
        data: np.ndarray,
        *,
        mu_start: float = None) -> float:
    mu_est = 1.0 if mu_start is None else mu_start
    return mu_est

def main(test_run:bool, output_folder:str="../../../output"):

    #_show_log_gamma()
    #_show_poisson()
    #_show_neg_log_poisson()
    #_show_negative_binomial()
    _show_neg_log_negative_binomial()

    poisson_to_gaussian = Poisson_to_Gaussian(1.0, # max_mu
                                              10, # n_mu_values
                                              5.0, # max_x_mu_scale
                                              1000) # n_x_values
    poisson_to_gaussian.build_inv_array()
    poisson_vals = []
    gaussian_vals = []
    mu = 0.1
    tot = 0.0
    n_vals = 100000
    for i in range(n_vals):
        rval01 = np.random.rand()
        #print("rval01=",rval01)
        poival = max(0.0, np.random.poisson(lam=mu)) # + rval01 - 0.5)
        #poival = scipy.stats.poisson.rvs(mu)
        poisson_vals.append(poival)
        #print("mu=",mu,"poival=",poival)
        gaussian_vals.append(poisson_to_gaussian.gaussian_value(mu, poival))
        #print("poival=",poival,"gval=",poisson_to_gaussian.gaussian_value(mu, poival))
        tot += poival

    print("Mean=",tot/n_vals)

    counts,bins = np.histogram(poisson_vals, bins=20, range=(0.0,3.0*mu))
    print("counts=",counts)
    plt.close("all")
    plt.figure(num=1, dpi=240)
    plt.plot(
        list(range(len(counts))),
        counts,
        marker="o",
    )
    
    plt.show()            

    counts,bins = np.histogram(gaussian_vals, bins=20, range=(-3.0*math.sqrt(mu),3.0*math.sqrt(mu)))
    plt.close("all")
    plt.figure(num=1, dpi=240)
    plt.plot(
        list(range(len(counts))),
        counts,
        marker="o",
    )
    
    plt.show()            

    np.random.seed(0) # We want the numbers to be the same on each run

    # data generation
    mu_pop = 5.0 # population distribution mean
    n_good_points = 100
    n_bad_points = 0
    x_min = 0.0
    x_max = 10.0
    data = np.zeros((n_good_points+n_bad_points,1))
    print(data.shape)
    dg = scipy.stats.poisson.rvs(mu_pop, size=n_good_points)
    print("dg=",dg)
    for i in range(n_good_points):
        #print(data[i],data[i][0])
        data[i][0] = dg[i]

    for i in range(n_good_points,n_good_points+n_bad_points):
        pass

    mu_est = _fit_poisson(data, mu_start=mu_pop+0.2)
    print("mu_est=",mu_est,"mu_pop=",mu_pop)

if __name__ == "__main__":
    main(False) # test_run
    
