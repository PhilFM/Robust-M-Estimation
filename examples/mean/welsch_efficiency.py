import numpy as np
import random
import math
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    import sys
    sys.path.append("../../pypi_package/src")

from gnc_smoothie_philfm.irls import IRLS
from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.gnc_null_params import GNC_NullParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc

from gncs_robust_mean import RobustMean

def main(test_run:bool, output_folder:str="../../output", quick_run:bool=False):
    sigma_base = 1.0
    sigma_limit = 500.0
    num_sigma_steps = 100
    max_niterations = 200
    small_val = 1.e-10

    def small_mean(optimiser_instance):
        a,AlB = optimiser_instance.weighted_derivs([0.0], 1.0) # model, lambda_val
        return -a[0]/AlB[0][0]

    def efficiency_est_func(p):
        return math.pow(1.0 + 2.0*p*p, 1.5)*math.pow(1.0 + p*p, -3.0)

    def efficiency_est_func_n(p,n):
        numerator = n*math.pow(1.0 + 2.0*p*p, -1.5)
        denom1 = (1.0 + 3.0*p*p*p*p + 2.0*p*p)*math.pow(1.0 + 2.0*p*p, -2.5)
        denom2 = (n-1.0)*math.pow(1.0 + p*p, -3.0)
        return (denom1 + denom2)/numerator

    for test_idx in range(0):
        n = 100
        data = np.zeros((n,1))
        weight = np.zeros(n)
        sigma_pop = 0.4
        for i in range(n):
            data[i][0] = random.gauss(0.0, sigma_pop)
            weight[i] = 1.0

        param_instance = GNC_WelschParams(WelschInfluenceFunc(),
                                          sigma_base, sigma_limit, num_sigma_steps, max_niterations=max_niterations)
        optimiser_instance = IRLS(param_instance, RobustMean(), data, weight, max_niterations=max_niterations)
        if optimiser_instance.run():
            m1 = optimiser_instance.final_model

        m2 = small_mean(param_instance.influence_func_instance)

        if not test_run:
            print("m1=",m1," m2=",m2)

    plt.close("all")
    plt.figure(num=1, dpi=240)
    ax = plt.gca()
    ax.set_xlabel(r'$p$')
    ax.set_ylabel('Efficiency')

    pmax = 1.0
    splist = np.linspace(0, pmax, num=30)

    n_samples = 30 if quick_run else 3000

    n_array = [5,10,50,100]
    col_array = ['magenta','r','g','cyan']
    for n,col in zip(n_array,col_array, strict=True):
        effData = []
        for sigma_pop in splist:
            mstot = 0.0
            lsstot = 0.0
            semx2s2 = 0.0
            sx2emx2s2 = 0.0
            sx4emx2s2 = 0.0
            semx22s2 = 0.0
            sx2emx22s2 = 0.0
            inv_variance = 1.0/(sigma_base*sigma_base)
            small_mean_var = 0.0
            small_mean_var_est = 0.0
            small_mean_var_num_est = 0.0
            small_mean_var_den_est = 0.0
            for test_idx in range(n_samples):
                data = np.zeros((n,1))
                weight = np.zeros(n)
                for i in range(n):
                    data[i] = [random.gauss(0.0, sigma_pop)]
                    weight[i] = 1.0

                influence_func_instance = WelschInfluenceFunc(sigma=sigma_base)
                param_instance = GNC_NullParams(influence_func_instance)
                optimiser_instance = SupGaussNewton(param_instance, RobustMean(), data, weight=weight, max_niterations=200)

                m = small_mean(optimiser_instance)
                mstot += m*m

                lsm = optimiser_instance.weighted_fit()[0]
                lsstot += lsm*lsm

                sxemx22s2 = 0.0
                sfid = 0.0
                for i in range(n):
                    x = data[i][0]
                    semx2s2 += math.exp(-x*x*inv_variance)
                    sx2emx2s2 += x*x*math.exp(-x*x*inv_variance)
                    sx4emx2s2 += x*x*x*x*math.exp(-x*x*inv_variance)
                    semx22s2 += math.exp(-0.5*x*x*inv_variance)
                    sx2emx22s2 += x*x*math.exp(-0.5*x*x*inv_variance)

                    sxemx22s2 += x*math.exp(-0.5*x*x*inv_variance)
                    sfid += (1.0 - x*x*inv_variance)*math.exp(-0.5*x*x*inv_variance)

                m2 = small_mean(optimiser_instance)
                #print("m2=",m2," m2p=",sxemx22s2/sfid)
                small_mean_var += m2*m2
                small_mean_var_est += math.pow(sxemx22s2/sfid, 2.0)
                small_mean_var_num_est += sxemx22s2*sxemx22s2
                small_mean_var_den_est += sfid*sfid

            semx2s2 /= n_samples*n
            sx2emx2s2 /= n_samples*n
            sx4emx2s2 /= n_samples*n
            semx22s2 /= n_samples*n
            sx2emx22s2 /= n_samples*n
            small_mean_var /= n_samples
            small_mean_var_est /= n_samples
            small_mean_var_num_est /= n_samples
            small_mean_var_den_est /= n_samples

            p = sigma_pop/sigma_base
            var = n*mstot/n_samples
            lsvar = n*lsstot/n_samples
            if not test_run:
                print("sigma_pop=",sigma_pop," var=",var, " est=",sigma_base*sigma_base*p*p*math.sqrt(1.0+p*p)," lsvar",lsvar," est=",sigma_base*sigma_base*p*p)

            effData.append((lsvar+small_val)/(var+small_val))

            semx2s2_est = math.pow(1.0+2.*p*p, -0.5)
            sx2emx2s2_est = p*p*math.pow(1.0+2.*p*p, -1.5)*sigma_base*sigma_base
            sx4emx2s2_est = 3.0*p*p*p*p*math.pow(1.0+2.*p*p, -2.5)*sigma_base*sigma_base*sigma_base*sigma_base
            semx22s2_est = math.pow(1.0+p*p, -0.5)
            sx2emx22s2_est = p*p*math.pow(1.0+p*p, -1.5)*sigma_base*sigma_base
            if not test_run:
                print("E(e^(-x^2/s^2)) = ", semx2s2_est, " est = ", semx2s2, " ratio = ", semx2s2_est/semx2s2)
                print("E(x^2*e^(-x^2/s^2)) = ", sx2emx2s2_est, " est = ", sx2emx2s2, " ratio = ", sx2emx2s2_est/sx2emx2s2)
                print("E(x^4*e^(-x^2/s^2)) = ", sx4emx2s2_est, " est = ", sx4emx2s2, " ratio = ", sx4emx2s2_est/sx4emx2s2)
                print("E(e^(-x^2/(2*s^2))) = ", semx22s2_est, " est = ", semx22s2, " ratio = ", semx22s2_est/semx22s2)
                print("E(x^2*e^(-x^2/(2*s^2))) = ", sx2emx22s2_est, " est = ", sx2emx22s2, " ratio = ", sx2emx22s2_est/sx2emx22s2)

            # calculate asymptotic efficiency
            numerator = n*sx2emx2s2_est
            denom1 = n*(semx2s2_est + sx4emx2s2_est*inv_variance*inv_variance - 2.0*sx2emx2s2_est*inv_variance)
            denom2 = n*(n-1.0)*(semx22s2_est*semx22s2_est + sx2emx22s2_est*sx2emx22s2_est*inv_variance*inv_variance - 2.0*sx2emx22s2_est*semx22s2_est*inv_variance)
            if not test_run:
                print("num=",numerator," den1=",denom1," den2=",denom2)

            numeratorp = p*p*math.pow(1.0 + 2.0*p*p, -1.5)
            denom1p = (1.0 + 3.0*p*p*p*p + 2.0*p*p)*math.pow(1.0 + 2.0*p*p, -2.5)
            denom2p = (n-1.0)*math.pow(1.0 + p*p, -3.0)
            if not test_run:
                print("nump=",n*numeratorp," den1p=",n*denom1p," den2p=",n*denom2p)

            small_mean_var_est2 = numerator/(denom1+denom2)
            small_mean_var_est3 = numeratorp/(denom1p+denom2p)
            small_mean_var_est4 = small_mean_var_num_est/small_mean_var_den_est
            if not test_run:
                print("Small mean variance = ", n*small_mean_var, " est = ", n*small_mean_var_est, " est2 = ", n*small_mean_var_est2, " est3 = ", n*small_mean_var_est3, " est4 = ", n*small_mean_var_est4)
                print("Ratio: ", small_mean_var_est2/small_mean_var_est, " num_est=", small_mean_var_num_est," den_est=", small_mean_var_den_est)
                print("p=",p, " asymptotic efficiency=", (small_val + sigma_pop*sigma_pop/n)/(small_val + small_mean_var_est2), " est=", efficiency_est_func(p), " estn=", efficiency_est_func_n(p,n))

        plt.plot(splist, effData, color = col, lw = 1.0, label = '$n=$' + str(n), marker = 'o', markersize = 2.0)
        #hmfv = np.vectorize(efficiency_est_func_n, excluded={"n"})
        #mlist = np.linspace(0, pmax, num=300)
        #plt.plot(mlist, hmfv(mlist, n=n), color = col, lw = 1.0, linestyle = 'dashed', )

    hmfv = np.vectorize(efficiency_est_func)
    mlist = np.linspace(0, pmax, num=300)
    plt.plot(mlist, hmfv(mlist), color = 'b', lw = 1.0, linestyle = 'dashed', label = 'asymptotic')

    plt.legend()
    plt.savefig(os.path.join(output_folder, "welsch_efficiency.png"), bbox_inches='tight')
    if not test_run:
        plt.show()

    if test_run:
        print("welsch_efficiency OK")

if __name__ == "__main__":
    main(False) # test_run
