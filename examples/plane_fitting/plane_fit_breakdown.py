import math
import numpy as np
import os
import sys

if __name__ == "__main__":
    sys.path.append("../../pypi_package/src")
    sys.path.append("../../pypi_package/src/gnc_smoothie_philfm/linear_model")
    sys.path.append("../../pypi_package/src/gnc_smoothie_philfm/cython")

from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie_philfm.linear_model.linear_regressor import LinearRegressor

sys.path.append("../misc")
from check_for_breakdown import check_for_breakdown

def next_bad_point(idx, n_points_xy):
    # check quadrant
    if idx[0] >= idx[1] and n_points_xy-idx[0]-1 > idx[1]: # lower quadrant
        #print("lower")
        idx[0] += 1
    elif idx[0] > idx[1] and n_points_xy-idx[0]-1 <= idx[1]: # right quadrant
        #print("right")
        idx[1] += 1
    elif idx[0] <= idx[1] and n_points_xy-idx[0]-1 >= idx[1]: # left quadrant
        #print("left")
        idx[1] -= 1
        if idx[1] <= idx[0]:
            idx[0] += 1
            idx[1] += 1
    else: # top quadrant
        #print("top")
        idx[0] -= 1

    return idx

def main(test_run:bool, output_folder:str="../../output", quick_run:bool=False):
    np.random.seed(0) # We want the numbers to be the same on each run

    # check out spiral sampling
    if False: #not test_run:
        n_points_xy = 10
        n_points = n_points_xy*n_points_xy
        bad_xy_idx = [0,0]
        for i in range(n_points):
            print("bad x,y=",bad_xy_idx)
            bad_xy_idx = next_bad_point(bad_xy_idx, n_points_xy)

    all_good = True
    outlier_ratio = 0.2
    n_points_xy = 4 if quick_run else 10
    n_points = n_points_xy*n_points_xy
    n_bad_points = int(outlier_ratio*n_points)
    #print("n_bad_points=",n_bad_points)
    sigma_pop = 0.3
    p = 0.666667
    sigma_base = sigma_pop/p

    # zero of second derivative of erf(x)/x, calculated using Brent algorithm in misc/erf_check.py
    erf_div_2nd_deriv_zero_point = 0.9678571637866076

    # f(a) = integral_{-D}^D exp*((a*x)^2/(2*sigma^2)) dx
    #      = pi*erf(a')/(D*a'), a' = a*D/(sqrt(2)*sigma)
    # So when a' == erf_div_2nd_deriv_zero_point, a = sqrt(2)*sigma*a'/D
    coeff_ab = erf_div_2nd_deriv_zero_point*math.sqrt(2.0)

    Dlist = (2.0,3.0) if quick_run else (1.0,2.0,3.0,4.0,5.0)
    for D in Dlist:
        if not test_run:
            print("D=",D)

        Dx = Dy = D
        data = np.zeros((n_points,3))
        for i in range(n_points_xy):
            y = -Dy + i*2.0*Dy/(n_points_xy-1)
            for j in range(n_points_xy):
                x = -Dx + j*2.0*Dx/(n_points_xy-1)
                idx = i*n_points_xy+j
                data[idx][0] = x
                data[idx][1] = y
                data[idx][2] = np.random.normal(0.0,sigma_pop) # good plane is a=b=0

        for a_idx in range(1 if quick_run else 10):
            if not test_run:
                print("a_idx=",a_idx)

            plane_bad = [0.5 + 0.1*a_idx, 0.0, 0.0]
            # pollute bad points
            bad_xy_idx = [0,0]
            for i in range(n_bad_points):
                idx = bad_xy_idx[1]*n_points_xy+bad_xy_idx[0]
                data[idx][2] = plane_bad[0]*data[idx][0] + plane_bad[1]*data[idx][1] + plane_bad[2]
                bad_xy_idx = next_bad_point(bad_xy_idx, n_points_xy)

            idx_bl = 0
            idx_br = n_points_xy-1
            idx_tl = (n_points_xy-1)*n_points_xy
            idx_tr = (n_points_xy-1)*n_points_xy + n_points_xy-1
            sigma_limit = abs(max(Dx,Dy,
                                  plane_bad[0]*data[idx_bl][0] + plane_bad[1]*data[idx_bl][1] + plane_bad[2],
                                  plane_bad[0]*data[idx_br][0] + plane_bad[1]*data[idx_br][1] + plane_bad[2],
                                  plane_bad[0]*data[idx_tl][0] + plane_bad[1]*data[idx_tl][1] + plane_bad[2],
                                  plane_bad[0]*data[idx_tr][0] + plane_bad[1]*data[idx_tr][1] + plane_bad[2]))
            num_sigma_steps = 3 if quick_run else 20
            plane_good = [0.0, 0.0, 0.0]

            influence_func = WelschInfluenceFunc()
            model_instance = LinearRegressor(data[0])
            param_instance = GNC_WelschParams(influence_func, sigma_base, sigma_limit, num_sigma_steps)
            optimiser_instance = SupGaussNewton(param_instance, data, model_instance=model_instance, model_start=plane_good)

            def plane_a(a, data):
                return optimiser_instance.objective_func([a,plane_good[1],plane_good[2]])

            func_a_v = np.vectorize(plane_a, excluded={"data"})
            param_instance.reset()
            for step in range(1+param_instance.n_steps()):
                all_good = check_for_breakdown(max(1.5,2.0*coeff_ab*influence_func.sigma/Dx, 1.5*abs(plane_bad[0])), func_a_v, data,
                                               coeff_ab*influence_func.sigma/Dx, # optimised but don't know what it means
                                               os.path.join(output_folder, "plane_fit_breakdown_a_D=" + str(int(D)) + "-i=" + str(a_idx) + "-" + str(step) + ".png"),
                                               "a", "a,0,0", test_run, all_good)
                param_instance.increment()

        for b_idx in range(1 if quick_run else 10):
            if not test_run:
                print("b_idx=",b_idx)

            plane_bad = [0.0, 0.5 + 0.1*b_idx, 0.0]

            # pollute bad points
            bad_xy_idx = [0,0]
            for i in range(n_bad_points):
                idx = bad_xy_idx[1]*n_points_xy+bad_xy_idx[0]
                data[idx][2] = plane_bad[0]*data[idx][0] + plane_bad[1]*data[idx][1] + plane_bad[2]
                bad_xy_idx = next_bad_point(bad_xy_idx, n_points_xy)

            def plane_b(b, data):
                return optimiser_instance.objective_func([plane_good[0],b,plane_good[2]])

            func_b_v = np.vectorize(plane_b, excluded={"data"})
            param_instance.reset()
            for step in range(1+param_instance.n_steps()):
                all_good = check_for_breakdown(max(1.5,2.0*coeff_ab*influence_func.sigma/Dy, 1.5*abs(plane_bad[1])), func_b_v, data,
                                               coeff_ab*influence_func.sigma/Dy,
                                               os.path.join(output_folder, "plane_fit_breakdown_b_D=" + str(int(D)) + "-i=" + str(b_idx) + "-" + str(step) + ".png"),
                                               "b", "0,b,0", test_run, all_good)
                param_instance.increment()

        for c_idx in range(1 if quick_run else 10):
            if not test_run:
                print("c_idx=",c_idx)

            plane_bad = [0.0, 0.0, sigma_pop*(1+b_idx)]

            # pollute bad points
            bad_xy_idx = [0,0]
            for i in range(n_bad_points):
                idx = bad_xy_idx[1]*n_points_xy+bad_xy_idx[0]
                data[idx][2] = plane_bad[0]*data[idx][0] + plane_bad[1]*data[idx][1] + plane_bad[2]
                bad_xy_idx = next_bad_point(bad_xy_idx, n_points_xy)

            def plane_c(c, data):
                return optimiser_instance.objective_func([plane_good[0],plane_good[2],c])

            func_c_v = np.vectorize(plane_c, excluded={"data"})
            param_instance.reset()
            for step in range(1+param_instance.n_steps()):
                all_good = check_for_breakdown(max(5,2.0*influence_func.sigma,1.5*abs(plane_bad[2])), func_c_v, data,
                                               influence_func.sigma,
                                               os.path.join(output_folder, "plane_fit_breakdown_c_D=" + str(int(D)) + "-i=" + str(c_idx) + "-" + str(step) + ".png"),
                                               "b", "0,b,0", test_run, all_good)
                param_instance.increment()

    if not test_run:
        print("Breakdown point threshold exceeded: ", not all_good)

    if test_run:
        print("plane_fit_breakdown OK")

if __name__ == "__main__":
    main(False) # test_run
