import math
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    import sys
    sys.path.append("../../pypi_package/src")

from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc

from line_fit import LineFit

def main(test_run:bool, output_folder:str="../../output"):
    np.random.seed(0) # We want the numbers to be the same on each run

    line_bad = [1.1, 0.0]
    n_points = 100
    outlier_ratio = 0.22
    n_bad_points = int(outlier_ratio*n_points)
    print("n_bad_points=",n_bad_points)
    D = 1.0
    sigma_pop = 0.3
    data = np.zeros((n_points,2))
    for i in range(n_points):
        x = -D + i*2.0*D/(n_points-1)
        data[i][0] = x
        data[i][1] = np.random.normal(0.0,sigma_pop) # good line is a=b=0

    # pollute bad points
    for i in range(n_bad_points//2):
        data[i][1] = line_bad[0]*data[i][0] + line_bad[1]
        ip = n_points-i-1
        data[ip][1] = line_bad[0]*data[ip][0] + line_bad[1]

    p = 0.666667
    sigma_base = sigma_pop/p
    sigma_limit = abs(max(D,line_bad[0]*data[0][0] + line_bad[1], line_bad[0]*data[len(data)-1][0] + line_bad[1]))
    num_sigma_steps = 20
    line_good = [0.0, 0.0]

    influence_func = WelschInfluenceFunc()
    line_fit = LineFit()
    param_instance = GNC_WelschParams(influence_func, sigma_base, sigma_limit, num_sigma_steps)
    optimiser_instance = SupGaussNewton(param_instance, line_fit, data, model_start=line_good)

    def line_a(a, data):
        return optimiser_instance.objective_func([a,line_good[1]])

    def line_b(b, data):
        return optimiser_instance.objective_func([line_good[0],b])

    # zero of second derivative of erf(x)/x, calculated using Brent algorithm in misc/erf_check.py
    erf_div_2nd_deriv_zero_point = 0.9678571637866076

    # f(a) = integral_{-D}^D exp*((a*x)^2/(2*sigma)) dx
    #      = pi*erf(a')/(D*a'), a' = a*D/(sqrt(2)*sigma)
    # So when a' == erf_div_2nd_deriv_zero_point, a = sqrt(2)*sigma*a'/D
    coeff_a = erf_div_2nd_deriv_zero_point*math.sqrt(2.0)

    func_a_v = np.vectorize(line_a, excluded={"data"})
    param_instance.reset()
    for step in range(1+param_instance.n_steps()):
        a_max = max(1.5,2.0*coeff_a*influence_func.sigma/D, 1.5*abs(line_bad[0]))
        alist = np.linspace(-a_max, a_max, num=400)

        #print(data)
        plt.close("all")
        plt.figure(num=1, dpi=240)
        ax = plt.gca()
        ax.set_xlabel(r"$a$")
        ax.set_ylabel(r"$F(a,0)$")

        a_tot_list = func_a_v(alist,data=data)
        y_max_idx = np.argmax(a_tot_list)
        y_max_a = alist[y_max_idx]
        y_max = 1.05*a_tot_list[y_max_idx]
        ax.set_ylim((0.0, y_max))
        plt.plot(alist, a_tot_list, lw = 1.0, color = 'green')

        breakdown_thres_a = coeff_a*influence_func.sigma/D # optimised but don't know what it means
        plt.axvline(x = -breakdown_thres_a, color = 'b', ymax = y_max, lw = 1.0)
        plt.axvline(x =  breakdown_thres_a, color = 'b', ymax = y_max, lw = 1.0)
        plt.axvline(x =                  0, color = 'r', ymax = y_max, lw = 1.0)
        plt.axvline(x =   y_max_a, color = 'cyan', ymax = y_max, lw = 1.0)

        #plt.legend()
        plt.savefig(os.path.join(output_folder, "line_fit_breakdown_a" + str(step) + ".png"), bbox_inches='tight')
        if abs(y_max_a) > breakdown_thres_a:
            if not test_run:
                plt.show()

        param_instance.increment()

    func_b_v = np.vectorize(line_b, excluded={"data"})
    param_instance.reset()
    for step in range(1+param_instance.n_steps()):
        b_max = max(5,2.0*influence_func.sigma,1.5*abs(line_bad[1]))
        blist = np.linspace(-b_max, b_max, num=400)

        #print(data)
        plt.close("all")
        plt.figure(num=1, dpi=240)
        ax = plt.gca()
        ax.set_xlabel(r"$b$")
        ax.set_ylabel(r"$F(0,b)$")

        b_tot_list = func_b_v(blist,data=data)
        y_max_idx = np.argmax(b_tot_list)
        y_max_b = blist[y_max_idx]
        y_max = 1.05*b_tot_list[y_max_idx]
        ax.set_ylim((0.0, y_max))
        plt.plot(blist, b_tot_list, lw = 1.0, color = 'green')

        breakdown_thres_b = influence_func.sigma
        plt.axvline(x = -breakdown_thres_b, color = 'b', ymax = y_max, lw = 1.0)
        plt.axvline(x =  breakdown_thres_b, color = 'b', ymax = y_max, lw = 1.0)
        plt.axvline(x =                  0, color = 'r', ymax = y_max, lw = 1.0)
        plt.axvline(x =   y_max_b, color = 'cyan', ymax = y_max, lw = 1.0)

        #plt.legend()
        plt.savefig(os.path.join(output_folder, "line_fit_breakdown_b" + str(step) + ".png"), bbox_inches='tight')
        if abs(y_max_b) > breakdown_thres_b:
            if not test_run:
                plt.show()

        param_instance.increment()

    if test_run:
        print("line_fit_breakdown OK")

if __name__ == "__main__":
    main(False) # test_run
