import numpy as np
import matplotlib.pyplot as plt
import math
import os
import sys

if __name__ == "__main__":
    sys.path.append("../../pypi_package/src")

from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc

from trs import TRS

sys.path.append("../misc")
from check_for_breakdown import check_for_breakdown

def randomM11() -> float:
    return 2.0*(np.random.rand()-0.5)

def apply_trs(trs, d, sigma=0.0):
    return (trs[1]*d[0] - trs[0]*d[1] + trs[2] + np.random.normal(0.0,sigma),
            trs[0]*d[0] + trs[1]*d[1] + trs[3] + np.random.normal(0.0,sigma))
            
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

    image_width = 1000
    image_height = 1000
    half_image_width = 0.5*image_width
    half_image_height = 0.5*image_height
    outlier_fraction = 0.7
    n_points_xy = 4 if quick_run else 10
    n_points = n_points_xy*n_points_xy
    n_bad_points = int(outlier_fraction*n_points)

    # zero of second derivative of erf(x)/x, calculated using Brent algorithm in misc/erf_check.py
    erf_div_2nd_deriv_zero_point = 0.9678571637866076

    # f(s) = integral_{-h/2}^{h/2} integral_{-(s*x)^2}^{w/2} exp*((s*y)^2/(2*sigma)) exp*((s*x)^2/(2*sigma)) dx dy
    #      = integral_{-h/2}^{h/2} pi*exp*((s*y)^2/(2*sigma))*erf(s')/(D*a') dy, s' = s*(w/2)/(sqrt(2)*sigma)
    #      = pi*erf(s')/((w/2)*s') * integral_{-h/2}^{h/2} exp*((s*y)^2/(2*sigma)) dy
    #      = pi*erf(s')/((w/2)*s') * pi*erf(s'')/((w/2)*s''), s'' = s*(h/2)/(sqrt(2)*sigma)
    # If h(x) = g(x)^2, g(x) = erf(x)/x
    #    h'(x) = 2*g(x)*g'(x)
    #    h''(x) = 2*(g(x)*g''(x) + g'(x)^2)
    # We have
    #    g'(x) = 2*exp(-x*x)/(sqrt(math.pi)*x) - erf(x)/(x*x)
    #    g''(x) = 2*(erf(x) - 2*exp(-x*x)*x*(1 + x*x)/sqrt(math.pi))/(x*x*x)
    # So
    #    h''(x) = 2*(erf(x)/x * 2*(erf(x) - 2*exp(-x*x)*x*(1 + x*x)/sqrt(math.pi))/(x*x*x) + (2*exp(-x*x)/(sqrt(math.pi)*x) - erf(x)/(x*x))^2)
    #           = 2*(-4*erf(x)*exp(-x*x)*(1 + x*x)/sqrt(math.pi) ...
    # So when w==h=wh and s' == s'', we have
    # f(s) = (pi*erf(s')/((wh/2)*s'))^2
    # When s' == erf_div_2nd_deriv_zero_point, s = sqrt(2)*sigma*s'/(wh/2)
    coeff_sc = erf_div_2nd_deriv_zero_point*math.sqrt(2.0)

    all_good = True
    for test_idx in range(0,1):
        trs_gt = [0.0,1.0,0.0,0.0]
        sigma_pop = 2.0

        # model is s,c,tx,ty
        data = np.zeros((n_points,4))
        for i in range(n_points_xy):
            y = -half_image_height + i*image_height/(n_points_xy-1)
            for j in range(n_points_xy):
                x = -half_image_width + j*image_width/(n_points_xy-1)
                idx = i*n_points_xy+j
                data[idx][0] = x
                data[idx][1] = y
                (data[idx][2],data[idx][3]) = apply_trs(trs_gt, data[idx], 0.0) #sigma_pop)

        # pollute bad points
        angle_bad = 0 #0.1*np.pi #*np.random.rand()
        scale_bad = 1 #0.95 #1.0 + 0.2*randomM11()
        s_bad = scale_bad*math.sin(angle_bad)
        c_bad = scale_bad*math.cos(angle_bad)
        tx_bad = 10.0*sigma_pop

        #if not test_run:
        #    print("data=",data)

        p = 0.66667
        sigma_base = sigma_pop/p
        sigma_limit = image_width
        num_sigma_steps = 20

        influence_func = WelschInfluenceFunc()
        param_instance = GNC_WelschParams(influence_func, sigma_base, sigma_limit, num_sigma_steps)
        optimiser_instance = SupGaussNewton(param_instance, TRS(), data)

        trs_bad = [s_bad, 0.0, 0.0, 0.0] #150.0*randomM11(), 70.0*randomM11()]
        bad_xy_idx = [0,0]
        for i in range(n_bad_points):
            idx = bad_xy_idx[1]*n_points_xy+bad_xy_idx[0]
            #print("idx=",idx)
            (data[idx][2],data[idx][3]) = apply_trs(trs_bad, data[idx], 0.0)
            bad_xy_idx = next_bad_point(bad_xy_idx, n_points_xy)

        def trs_s(s, data):
            return math.sqrt(optimiser_instance.objective_func([s, trs_gt[1], trs_gt[2], trs_gt[3]]))

        func_s_v = np.vectorize(trs_s, excluded={"data"})
        param_instance.reset()
        for step in range(1+param_instance.n_steps()):
            breakdown_thres_s = coeff_sc*influence_func.sigma/half_image_width
            all_good = check_for_breakdown(max(0.02, 2.0*breakdown_thres_s, 1.5*abs(trs_bad[0])), func_s_v, data,
                                           breakdown_thres_s,
                                           os.path.join(output_folder, "trs_breakdown_s_" + str(step) + ".png"),
                                           "s", "s,0,0,0", test_run, all_good)
            param_instance.increment()

        trs_bad = [0, c_bad, 0.0, 0.0] #150.0*randomM11(), 70.0*randomM11()]
        bad_xy_idx = [0,0]
        for i in range(n_bad_points):
            idx = bad_xy_idx[1]*n_points_xy+bad_xy_idx[0]
            #print("idx=",idx)
            (data[idx][2],data[idx][3]) = apply_trs(trs_bad, data[idx], 0.0)
            bad_xy_idx = next_bad_point(bad_xy_idx, n_points_xy)

        def trs_c(c, data):
            return math.sqrt(optimiser_instance.objective_func([trs_gt[0], 1.0+c, trs_gt[2], trs_gt[3]]))

        func_c_v = np.vectorize(trs_c, excluded={"data"})
        param_instance.reset()
        for step in range(1+param_instance.n_steps()):
            breakdown_thres_c = coeff_sc*influence_func.sigma/half_image_width
            all_good = check_for_breakdown(max(0.02, 2.0*breakdown_thres_c, 1.5*abs(trs_bad[1]-1.0)), func_c_v, data,
                                           breakdown_thres_c,
                                           os.path.join(output_folder, "trs_breakdown_c_" + str(step) + ".png"),
                                           "c", "0,c,0,0", test_run, all_good)
            param_instance.increment()

        trs_bad = [0, 0, tx_bad, 0.0] #150.0*randomM11(), 70.0*randomM11()]
        bad_xy_idx = [0,0]
        for i in range(n_bad_points):
            idx = bad_xy_idx[1]*n_points_xy+bad_xy_idx[0]
            #print("idx=",idx)
            (data[idx][2],data[idx][3]) = apply_trs(trs_bad, data[idx], 0.0)
            bad_xy_idx = next_bad_point(bad_xy_idx, n_points_xy)

        def trs_tx(tx, data):
            return optimiser_instance.objective_func([trs_gt[0], trs_gt[1], tx, trs_gt[3]])

        func_tx_v = np.vectorize(trs_tx, excluded={"data"})
        param_instance.reset()
        for step in range(1+param_instance.n_steps()):
            breakdown_thres_tx = influence_func.sigma
            all_good = check_for_breakdown(max(0.02, 2.0*breakdown_thres_tx, 1.5*abs(trs_bad[2])), func_tx_v, data,
                                           breakdown_thres_tx,
                                           os.path.join(output_folder, "trs_breakdown_tx_" + str(step) + ".png"),
                                           "tx", "0,0,tx,0", test_run, all_good)
            param_instance.increment()

        #def trs_ty(ty, data):
        #    return optimiser_instance.objective_func([trs_gt[0], trs_gt[1], trs_gt[2], ty])

    if not test_run:
        print("Breakdown point threshold exceeded: ", not all_good)

    if test_run:
        print("trs_breakdown OK")

if __name__ == "__main__":
    main(False) # test_run
