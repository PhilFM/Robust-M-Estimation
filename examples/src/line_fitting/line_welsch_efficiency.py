import numpy as np
import random
import math
import matplotlib.pyplot as plt
import os
import sys

if __name__ == "__main__":
    sys.path.append("../../../pypi_package/src")

from gnc_smoothie.sup_gauss_newton import SupGaussNewton
from gnc_smoothie.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie.cython_files.linear_regressor_welsch_evaluator import LinearRegressorWelschEvaluator

def main(test_run:bool, output_folder:str="../../../output", quick_run:bool=False):
    random.seed(321) # ensure results are the same on every run

    D = 5.0
    sigma_base = 0.01
    sigma_limit = sigma_base
    num_sigma_steps = 1
    max_niterations = 200
    small_val = 1.e-20

    def small_line(optimiser_instance, data, weight):
        if not optimiser_instance.fit(data, weight):
            param_instance_test = GNC_WelschParams(WelschInfluenceFunc(), sigma_base)
            optimiser_instance_debug = SupGaussNewton(param_instance_test, evaluator_instance=LinearRegressorWelschEvaluator(optimiser_instance._data[0][0]), max_niterations=max_niterations, messages_file=sys.stdout, debug=True)
            optimiser_instance_debug.fit(data, weight=weight)
            assert(False)

        return optimiser_instance.final_model

    def relative_efficiency_est_func(q):
        return math.pow(1.0 + 2.0*q*q, 1.5)*math.pow(1.0 + q*q, -3.0)

    def relative_efficiency_est_func_n(q:float, n:float):
        # these values are scaled
        numerator = math.pow(1.0+2.*q*q, -1.5)
        denom1 = 0.2*(1.0 + 3.0*q*q*q*q + 2.0*q*q)*math.pow(1.0 + 2.0*q*q, -2.5)
        denom2 = 0.11111111111*(n-1.0)*math.pow(1.0 + q*q, -3.0)
        ls_var = 1.0/(n*(0.2 + 0.1111111111*(n-1.0)))
        return ls_var*(denom1 + denom2)/numerator

    def least_squares_variance_a(sigma_pop: float, D: float, n: float):
        return 0.333333333333*sigma_pop*sigma_pop/(D*D*(0.2 + 0.1111111111*(n-1)))

    for test_idx in range(0):
        n = 10000
        data = np.zeros((n,2))
        weight = np.ones(n)
        sigma_pop = 0.4
        for i in range(n):
            data[i][0] = -D + 2.0*D*i/(n-1)
            data[i][1] = random.gauss(0.0, sigma_pop)

        param_instance = GNC_WelschParams(WelschInfluenceFunc(), sigma_base, sigma_limit=sigma_limit, num_sigma_steps=num_sigma_steps)
        optimiser_instance = SupGaussNewton(param_instance, evaluator_instance=LinearRegressorWelschEvaluator(data[0]), max_niterations=max_niterations)
        if optimiser_instance.fit(data, weight=weight):
            line1 = optimiser_instance.final_model

        line2 = small_line(optimiser_instance, data, weight)

        if not test_run:
            print("line1=",line1," line2=",line2)

        if False:
            # check some calculations used later
            semy2 = 0.0
            sx4emy2s2 = 0.0
            inv_variance = 1.0/(sigma_base*sigma_base)
            for i in range(n):
                x = data[i][0]
                y = data[i][1]
                semy2 += math.exp(-0.5*inv_variance*y*y)
                sx4emy2s2 += x*x*x*x*math.exp(-y*y*inv_variance)

            q = sigma_pop/sigma_base
            if not test_run:
                print("Check equal",semy2/n, 1.0/math.sqrt(1.0 + q*q))
                print("Check equal2",sx4emy2s2/n, D*D*D*D*0.2/math.sqrt(1.0 + 2.0*q*q))

    if False:
        # calculate least-squares estimate
        n = 1000
        data = np.zeros((n,2))
        weight = np.ones(n)
        sigma_pop = 0.4
        n_samples = 20000
        a2_tot = 0.0
        Sxx2_av = 0.0
        Sxy2_av = 0.0
        for sample in range(n_samples):
            for i in range(n):
                data[i][0] = -D + 2.0*D*i/(n-1)
                data[i][1] = random.gauss(0.0, sigma_pop)

            Sx = Sy = Sxx = Sxy = 0.0
            for i in range(n):
                x = data[i][0]
                y = data[i][1]
                Sx += x
                Sy += y
                Sxx += x*x
                Sxy += x*y

            # A = (Sxx Sx) so det(A) = n*Sxx - Sx*Sx, inv(A) = (  n -Sx)
            #     (Sx   n)                                     (-Sx Sxx)/det
            det = Sxx*n - Sx*Sx
            a = (n*Sxy - Sx*Sy)/det
            a2_tot += a*a

            Sxx2_av += Sxx*Sxx
            Sxy2_av += Sxy*Sxy

        Sxx2_av /= n_samples
        Sxy2_av /= n_samples

        # check least-squares variance calculation
        Sxx2_est = D*D*D*D*(0.2 + 0.1111111111*(n-1.0))
        Sxy2_est = 0.3333333333*D*D*sigma_pop*sigma_pop
        if not test_run:
            print("Sxx^2/n=",Sxx2_av/n, "est=", Sxx2_est)
            print("Sxy^2/n=",Sxy2_av/n, "est=", Sxy2_est)
            print("Variance=", a2_tot/(n_samples-1.0), "estimates=",Sxy2_est/Sxx2_est,Sxy2_av/Sxx2_av,least_squares_variance_a(sigma_pop, D, n))

    plt.close("all")
    plt.figure(num=1, dpi=240)
    ax = plt.gca()
    ax.set_xlabel(r'$q$')
    ax.set_ylabel('Relative efficiency')

    qmax = 1.0
    qlist = np.linspace(0, qmax, num=5 if test_run else 30)

    n_samples = 30 if quick_run else 3000

    n_array = [5] if test_run else [20,50] #,100,500]
    col_array = ['r'] if test_run else ['magenta','r'] #,'g','cyan']
    mlist = np.linspace(0, qmax, num=300)
    for n,col in zip(n_array,col_array, strict=True):
        effData = []
        for q in qlist:
            sigma_pop = q*sigma_base
            lastot = 0.0
            lsstot = 0.0
            sx2y2emy2s2 = 0.0
            sx4emy2s2 = 0.0
            sx4y2emy2s2 = 0.0
            sx4y4emy2s2 = 0.0
            sx2emy22s2 = 0.0
            sx2y2emy22s2 = 0.0
            inv_variance = 1.0/(sigma_base*sigma_base)
            small_line_a_var = 0.0
            #small_line_a_var_est = 0.0
            #small_line_a_var_num_est = 0.0
            #small_line_a_var_den_est = 0.0
            for test_idx in range(n_samples):
                data = np.zeros((n,2))
                weight = np.ones(n)
                for i in range(n):
                    data[i][0] = -D + 2.0*D*i/(n-1)
                    data[i][1] = random.gauss(0.0, sigma_pop)

                #print("data=",data)
                param_instance = GNC_WelschParams(WelschInfluenceFunc(), sigma_base)
                optimiser_instance = SupGaussNewton(param_instance,
                                                    evaluator_instance=LinearRegressorWelschEvaluator(data[0]),
                                                    max_niterations=200)

                line = small_line(optimiser_instance, data, weight)
                lastot += line[0]*line[0]

                ls_a = optimiser_instance.weighted_fit()[0][0]
                lsstot += ls_a*ls_a
                #print("Line",1000000.0*line[0],"ls:",1000000.0*ls_a)

                # These values relate to 2nd derivatives, we haven't worked these out correctly so ignore for now
                #sx2yemy22s2 = 0.0
                #sfid = 0.0
                for i in range(n):
                    x = data[i][0]
                    y = data[i][1]
                    sx2y2emy2s2 += x*x*y*y*math.exp(-y*y*inv_variance)
                    sx4emy2s2 += x*x*x*x*math.exp(-y*y*inv_variance)
                    sx4y2emy2s2 += x*x*x*x*y*y*math.exp(-y*y*inv_variance)
                    sx4y4emy2s2 += x*x*x*x*y*y*y*y*math.exp(-y*y*inv_variance)
                    sx2emy22s2 += x*x*math.exp(-0.5*y*y*inv_variance)
                    sx2y2emy22s2 += x*x*y*y*math.exp(-0.5*y*y*inv_variance)

                    #sx2yemy22s2 += x*x*y*math.exp(-0.5*y*y*inv_variance)
                    #sfid += x*x*(1.0 - y*y*inv_variance)*math.exp(-0.5*y*y*inv_variance)

                line2 = small_line(optimiser_instance, data, weight)
                #print("m2=",m2," m2p=",sxemx22s2/sfid)
                small_line_a_var += line2[0]*line2[0]
                #small_line_a_var_est += math.pow(sx2yemy22s2/sfid, 2.0)
                #small_line_a_var_num_est += sx2yemy22s2*sx2yemy22s2
                #small_line_a_var_den_est += sfid*sfid

            sx2y2emy2s2 /= n_samples*n
            sx4emy2s2 /= n_samples*n
            sx4y2emy2s2 /= n_samples*n
            sx4y4emy2s2 /= n_samples*n
            sx2emy22s2 /= n_samples*n
            sx2y2emy22s2 /= n_samples*n
            small_line_a_var /= n_samples
            #small_line_a_var_est /= n_samples
            #small_line_a_var_num_est /= n_samples
            #small_line_a_var_den_est /= n_samples

            q = sigma_pop/sigma_base
            var = lastot/(n_samples-1)
            lsvar = lsstot/(n_samples-1)
            if not test_run:
                print("sigma_pop=",sigma_pop," var=",var, " est=",3.0*sigma_pop*sigma_pop*math.sqrt(1.0+q*q)/(n*D*D)," lsvar",lsvar," est=",3.0*sigma_pop*sigma_pop/(n*D*D))

            effData.append((lsvar+small_val)/(var+small_val))

            sx2y2emy2s2_est = D*D*0.3333333333*q*q*math.pow(1.0+2.*q*q, -1.5)*sigma_base*sigma_base
            sx4emy2s2_est = D*D*D*D*0.2/math.sqrt(1.0 + 2.0*q*q)
            sx4y2emy2s2_est = D*D*D*D*0.2*q*q*math.pow(1.0+2.*q*q, -1.5)*sigma_base*sigma_base
            sx4y4emy2s2_est = D*D*D*D*0.2*3.0*q*q*q*q*math.pow(1.0+2.*q*q, -2.5)*sigma_base*sigma_base*sigma_base*sigma_base
            sx2emy22s2_est = D*D*0.33333333*math.pow(1.0+q*q, -0.5)
            sx2y2emy22s2_est = D*D*0.33333333*q*q*math.pow(1.0+q*q, -1.5)*sigma_base*sigma_base
            if not test_run:
                print("E(x^2*y^2e^(-y^2/s^2)) = ", sx2y2emy2s2_est, " est = ", sx2y2emy2s2, " ratio = ", sx2y2emy2s2_est/sx2y2emy2s2)
                print("E(x^4*e^(-y^2/s^2)) = ", sx4emy2s2_est, " est = ", sx4emy2s2, " ratio = ", sx4emy2s2_est/sx4emy2s2)
                print("E(x^4*y^2e^(-y^2/s^2)) = ", sx4y2emy2s2_est, " est = ", sx4y2emy2s2, " ratio = ", sx4y2emy2s2_est/sx4y2emy2s2)
                print("E(x^4*y^4*e^(-y^2/s^2)) = ", sx4y4emy2s2_est, " est = ", sx4y4emy2s2, " ratio = ", sx4y4emy2s2_est/sx4y4emy2s2)
                print("E(x^2*e^(-y^2/(2*s^2))) = ", sx2emy22s2_est, " est = ", sx2emy22s2, " ratio = ", sx2emy22s2_est/sx2emy22s2)
                print("E(x^2*y^2*e^(-y^2/(2*s^2))) = ", sx2y2emy22s2_est, " est = ", sx2y2emy22s2, " ratio = ", sx2y2emy22s2_est/sx2y2emy22s2)

            # calculate asymptotic efficiency
            numerator = n*sx2y2emy2s2_est
            denom1 = n*(sx4emy2s2_est + sx4y4emy2s2_est*inv_variance*inv_variance - 2.0*sx4y2emy2s2_est*inv_variance)
            denom2 = n*(n-1.0)*(sx2emy22s2_est*sx2emy22s2_est + sx2y2emy22s2_est*sx2y2emy22s2_est*inv_variance*inv_variance - 2.0*sx2y2emy22s2_est*sx2emy22s2_est*inv_variance)
            if not test_run:
                print("num=",numerator," den1=",denom1," den2=",denom2)

            numeratorp = D*D*0.3333333333*math.pow(1.0+2.*q*q, -1.5)*sigma_pop*sigma_pop
            denom1p = D*D*D*D*0.2*(1.0 + 3.0*q*q*q*q + 2.0*q*q)*math.pow(1.0 + 2.0*q*q, -2.5)
            denom2p = 0.11111111111*D*D*D*D*(n-1.0)*math.pow(1.0 + q*q, -3.0)
            if not test_run:
                print("nump=",n*numeratorp," den1p=",n*denom1p," den2p=",n*denom2p)

            small_line_a_var_est2 = numerator/(denom1+denom2)
            #small_line_a_var_est3 = numeratorp/(denom1p+denom2p)
            #small_line_a_var_est4 = small_line_a_var_num_est/small_line_a_var_den_est
            if not test_run:
                #print("Small line a variance = ", n*small_line_a_var, " est = ", n*small_line_a_var_est, " est2 = ", n*small_line_a_var_est2, " est3 = ", n*small_line_a_var_est3) #, " est4 = ", n*small_line_a_var_est4)
                #print("Ratio: ", small_line_a_var_est2/small_line_a_var_est, " num_est=", small_line_a_var_num_est," den_est=", small_line_a_var_den_est)
                print("q=",q, " asymptotic relative efficiency=", (small_val + 3.0*sigma_pop*sigma_pop/(D*D*n))/(small_val + small_line_a_var_est2), " est=", relative_efficiency_est_func(q), " estn=", relative_efficiency_est_func_n(q,n))

        plt.plot(qlist, effData, color = col, lw = 1.0, label = '$n=$' + str(n), marker = 'o', markersize = 2.0)
        #hmfv = np.vectorize(relative_efficiency_est_func_n, excluded={"n"})
        #plt.plot(mlist, hmfv(mlist, n=n), color = col, lw = 1.0, linestyle = 'dashed', )

    ax.set_ylim((0.5, 1.1))
    ref_q = 0.6666667
    plt.axline((ref_q,0), (ref_q,1.1), color = "grey")
    plt.text(ref_q-0.01, 0.6, "q=0.667", color = "grey", verticalalignment="center", horizontalalignment="right")

    hmfv = np.vectorize(relative_efficiency_est_func)
    plt.plot(mlist, hmfv(mlist), color = 'b', lw = 1.0, linestyle = 'dashed', label = 'asymptotic')

    plt.legend()
    plt.savefig(os.path.join(output_folder, "line_welsch_efficiency.png"), bbox_inches='tight')
    if not test_run:
        plt.show()

    if test_run:
        print("line_welsch_efficiency OK")

if __name__ == "__main__":
    main(False) # test_run
