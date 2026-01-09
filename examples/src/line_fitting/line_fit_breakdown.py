import math
import numpy as np
import os
import sys

if __name__ == "__main__":
    sys.path.append("../../../pypi_package/src")

from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie_philfm.linear_model.linear_regressor import LinearRegressor

sys.path.append("../misc")
from check_for_breakdown import check_for_breakdown

def main(test_run:bool, output_folder:str="../../../output", quick_run:bool=False):
    np.random.seed(0) # We want the numbers to be the same on each run

    all_good = True
    outlier_ratio = 0.19
    n_points = 20 if quick_run else 100
    n_bad_points = int(outlier_ratio*n_points)
    #print("n_bad_points=",n_bad_points)
    sigma_pop = 0.3
    p = 0.666667
    sigma_base = sigma_pop/p
    sigma_limit = 5.0
    num_sigma_steps = 3 if quick_run else 20

    line_good = [0.0, 0.0]

    # zero of second derivative of erf(x)/x, calculated using Brent algorithm in misc/erf_check.py
    erf_div_2nd_deriv_zero_point = 0.9678571637866076

    # f(a) = integral_{-D}^D exp*((a*x)^2/(2*sigma^2)) dx
    #      = pi*erf(a')/(D*a'), a' = a*D/(sqrt(2)*sigma)
    # So when a' == erf_div_2nd_deriv_zero_point, a = sqrt(2)*sigma*a'/D
    coeff_a = erf_div_2nd_deriv_zero_point*math.sqrt(2.0)

    Dlist = (2.0,3.0) if quick_run else (1.0,2.0,3.0,4.0,5.0)
    for D in Dlist:
        if not test_run:
            print("D=",D)

        data = np.zeros((n_points,2))
        for i in range(n_points):
            x = -D + i*2.0*D/(n_points-1)
            data[i][0] = x
            data[i][1] = np.random.normal(0.0,sigma_pop) # good line is a=b=0

        influence_func = WelschInfluenceFunc()
        model_instance = LinearRegressor(data[0])
        param_instance = GNC_WelschParams(influence_func, sigma_base, sigma_limit, num_sigma_steps)
        optimiser_instance = SupGaussNewton(param_instance, data, model_instance=model_instance)

        for a_idx in range(2 if quick_run else 10):
            if not test_run:
                print("a_idx=",a_idx)

            # pollute bad points
            line_bad = [0.5 + 0.1*a_idx, 0.0]
            for i in range(n_bad_points//2):
                data[i][1] = line_bad[0]*data[i][0] + line_bad[1]
                ip = n_points-i-1
                data[ip][1] = line_bad[0]*data[ip][0] + line_bad[1]

            def line_a(a, data):
                return optimiser_instance.objective_func([a,line_good[1]])

            func_a_v = np.vectorize(line_a, excluded={"data"})
            param_instance.reset()
            for step in range(1+param_instance.n_steps()):
                all_good = check_for_breakdown(max(1.5,2.0*coeff_a*influence_func.sigma/D, 1.5*abs(line_bad[0])), func_a_v, data,
                                               coeff_a*influence_func.sigma/D, # optimised but don't know what it means
                                               os.path.join(output_folder, "line_fit_breakdown_a_D=" + str(int(D)) + "-i=" + str(a_idx) + "-" + str(step) + ".png"),
                                               "a", "a,0", test_run, all_good)
                param_instance.increment()

        for b_idx in range(2 if quick_run else 10):
            if not test_run:
                print("b_idx=",b_idx)

            # pollute bad points
            line_bad = [0.0, sigma_pop*(1+b_idx)]
            for i in range(n_bad_points//2):
                data[i][1] = line_bad[0]*data[i][0] + line_bad[1]
                ip = n_points-i-1
                data[ip][1] = line_bad[0]*data[ip][0] + line_bad[1]

            def line_b(b, data):
                return optimiser_instance.objective_func([line_good[0],b])

            func_b_v = np.vectorize(line_b, excluded={"data"})
            param_instance.reset()
            for step in range(1+param_instance.n_steps()):
                all_good = check_for_breakdown(max(5,2.0*influence_func.sigma,1.5*abs(line_bad[1])), func_b_v, data,
                                               influence_func.sigma, # breakdown_thres
                                               os.path.join(output_folder, "line_fit_breakdown_b_D=" + str(int(D)) + "-i=" + str(b_idx) + "-" + str(step) + ".png"),
                                               "b", "0,b", test_run, all_good)
                param_instance.increment()

    if not test_run:
        print("Breakdown point threshold exceeded: ", not all_good)

    if test_run:
        print("line_fit_breakdown OK")

if __name__ == "__main__":
    main(False) # test_run
