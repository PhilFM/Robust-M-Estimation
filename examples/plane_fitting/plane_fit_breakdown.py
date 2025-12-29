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

from plane_fit import PlaneFit

def next_bad_point(idx, n_points_xy):
    # check quadrant
    if idx[0] >= idx[1] and n_points_xy-idx[0]-1 > idx[1]: # lower quadrant
        print("lower")
        idx[0] += 1
    elif idx[0] > idx[1] and n_points_xy-idx[0]-1 <= idx[1]: # right quadrant
        print("right")
        idx[1] += 1
    elif idx[0] <= idx[1] and n_points_xy-idx[0]-1 >= idx[1]: # left quadrant
        print("left")
        idx[1] -= 1
        if idx[1] <= idx[0]:
            idx[0] += 1
            idx[1] += 1
    else: # top quadrant
        print("top")
        idx[0] -= 1

    return idx

def main(test_run:bool, output_folder:str="../../output"):
    np.random.seed(0) # We want the numbers to be the same on each run

    plane_bad = [0.0, 1.1, 0.0]
    n_points_xy = 10
    n_points = n_points_xy*n_points_xy
    outlier_ratio = 0.4
    n_bad_points = int(outlier_ratio*n_points)
    print("n_bad_points=",n_bad_points)
    Dx = 1.0
    Dy = 1.0
    sigma_pop = 0.3
    data = np.zeros((n_points,3))
    for i in range(n_points_xy):
        y = -Dy + i*2.0*Dy/(n_points_xy-1)
        for j in range(n_points_xy):
            x = -Dx + j*2.0*Dx/(n_points_xy-1)
            idx = i*n_points_xy+j
            data[idx][0] = x
            data[idx][1] = y
            data[idx][2] = np.random.normal(0.0,sigma_pop) # good plane is a=b=0

    bad_xy_idx = [0,0]
    for i in range(n_points):
        print("bad x,y=",bad_xy_idx)
        bad_xy_idx = next_bad_point(bad_xy_idx, n_points_xy)

    # pollute bad points
    bad_xy_idx = [0,0]
    for i in range(n_bad_points):
        idx = bad_xy_idx[1]*n_points_xy+bad_xy_idx[0]
        data[idx][2] = plane_bad[0]*data[idx][0] + plane_bad[1]*data[idx][1] + plane_bad[2]
        bad_xy_idx = next_bad_point(bad_xy_idx, n_points_xy)

    p = 0.666667
    sigma_base = sigma_pop/p
    idx_bl = 0
    idx_br = n_points_xy-1
    idx_tl = (n_points_xy-1)*n_points_xy
    idx_tr = (n_points_xy-1)*n_points_xy + n_points_xy-1
    sigma_limit = abs(max(Dx,Dy,
                          plane_bad[0]*data[idx_bl][0] + plane_bad[1]*data[idx_bl][1] + plane_bad[2],
                          plane_bad[0]*data[idx_br][0] + plane_bad[1]*data[idx_br][1] + plane_bad[2],
                          plane_bad[0]*data[idx_tl][0] + plane_bad[1]*data[idx_tl][1] + plane_bad[2],
                          plane_bad[0]*data[idx_tr][0] + plane_bad[1]*data[idx_tr][1] + plane_bad[2]))
    num_sigma_steps = 20
    plane_good = [0.0, 0.0, 0.0]

    influence_func = WelschInfluenceFunc()
    plane_fit = PlaneFit()
    param_instance = GNC_WelschParams(influence_func, sigma_base, sigma_limit, num_sigma_steps)
    optimiser_instance = SupGaussNewton(param_instance, plane_fit, data, model_start=plane_good)

    def plane_a(a, data):
        return optimiser_instance.objective_func([a,plane_good[1],plane_good[2]])

    def plane_b(b, data):
        return optimiser_instance.objective_func([plane_good[0],b,plane_good[2]])

    def plane_c(c, data):
        return optimiser_instance.objective_func([plane_good[0],plane_good[2],c])

    # zero of second derivative of erf(x)/x, calculated using Brent algorithm in misc/erf_check.py
    erf_div_2nd_deriv_zero_point = 0.9678571637866076

    # f(a) = integral_{-D}^D exp*((a*x)^2/(2*sigma)) dx
    #      = pi*erf(a')/(D*a'), a' = a*D/(sqrt(2)*sigma)
    # So when a' == erf_div_2nd_deriv_zero_point, a = sqrt(2)*sigma*a'/D
    coeff_ab = erf_div_2nd_deriv_zero_point*math.sqrt(2.0)

    func_a_v = np.vectorize(plane_a, excluded={"data"})
    param_instance.reset()
    for step in range(1+param_instance.n_steps()):
        a_max = max(1.5,2.0*coeff_ab*influence_func.sigma/Dx, 1.5*abs(plane_bad[0]))
        alist = np.linspace(-a_max, a_max, num=400)

        #print(data)
        plt.close("all")
        plt.figure(num=1, dpi=240)
        ax = plt.gca()
        ax.set_xlabel(r"$a$")
        ax.set_ylabel(r"$F(a,0,0)$")

        a_tot_list = func_a_v(alist,data=data)
        y_max_idx = np.argmax(a_tot_list)
        y_max_a = alist[y_max_idx]
        y_max = 1.05*a_tot_list[y_max_idx]
        ax.set_ylim((0.0, y_max))
        plt.plot(alist, a_tot_list, lw = 1.0, color = 'green')

        breakdown_thres_a = coeff_ab*influence_func.sigma/Dx # optimised but don't know what it means
        plt.axvline(x = -breakdown_thres_a, color = 'b', ymax = y_max, lw = 1.0)
        plt.axvline(x =  breakdown_thres_a, color = 'b', ymax = y_max, lw = 1.0)
        plt.axvline(x =                  0, color = 'r', ymax = y_max, lw = 1.0)
        plt.axvline(x =   y_max_a, color = 'cyan', ymax = y_max, lw = 1.0)

        #plt.legend()
        plt.savefig(os.path.join(output_folder, "plane_fit_breakdown_a" + str(step) + ".png"), bbox_inches='tight')
        if abs(y_max_a) > breakdown_thres_a:
            if not test_run:
                plt.show()

        param_instance.increment()

    func_b_v = np.vectorize(plane_b, excluded={"data"})
    param_instance.reset()
    for step in range(1+param_instance.n_steps()):
        b_max = max(1.5,2.0*coeff_ab*influence_func.sigma/Dy, 1.5*abs(plane_bad[1]))
        blist = np.linspace(-b_max, b_max, num=400)

        #print(data)
        plt.close("all")
        plt.figure(num=1, dpi=240)
        ax = plt.gca()
        ax.set_xlabel(r"$b$")
        ax.set_ylabel(r"$F(0,b,0)$")

        b_tot_list = func_b_v(blist,data=data)
        y_max_idx = np.argmax(b_tot_list)
        y_max_b = alist[y_max_idx]
        y_max = 1.05*b_tot_list[y_max_idx]
        ax.set_ylim((0.0, y_max))
        plt.plot(alist, b_tot_list, lw = 1.0, color = 'green')

        breakdown_thres_b = coeff_ab*influence_func.sigma/Dy # optimised but don't know what it means
        plt.axvline(x = -breakdown_thres_a, color = 'b', ymax = y_max, lw = 1.0)
        plt.axvline(x =  breakdown_thres_a, color = 'b', ymax = y_max, lw = 1.0)
        plt.axvline(x =                  0, color = 'r', ymax = y_max, lw = 1.0)
        plt.axvline(x =   y_max_b, color = 'cyan', ymax = y_max, lw = 1.0)

        #plt.legend()
        plt.savefig(os.path.join(output_folder, "plane_fit_breakdown_b" + str(step) + ".png"), bbox_inches='tight')
        if abs(y_max_b) > breakdown_thres_b:
            if not test_run:
                plt.show()

        param_instance.increment()

    func_c_v = np.vectorize(plane_c, excluded={"data"})
    param_instance.reset()
    for step in range(1+param_instance.n_steps()):
        c_max = max(5,2.0*influence_func.sigma,1.5*abs(plane_bad[2]))
        clist = np.linspace(-c_max, c_max, num=400)

        #print(data)
        plt.close("all")
        plt.figure(num=1, dpi=240)
        ax = plt.gca()
        ax.set_xlabel(r"$c$")
        ax.set_ylabel(r"$F(0,0,c)$")

        c_tot_list = func_c_v(clist,data=data)
        y_max_idx = np.argmax(c_tot_list)
        y_max_c = clist[y_max_idx]
        y_max = 1.05*c_tot_list[y_max_idx]
        ax.set_ylim((0.0, y_max))
        plt.plot(clist, c_tot_list, lw = 1.0, color = 'green')

        breakdown_thres_c = influence_func.sigma
        plt.axvline(x = -breakdown_thres_c, color = 'b', ymax = y_max, lw = 1.0)
        plt.axvline(x =  breakdown_thres_c, color = 'b', ymax = y_max, lw = 1.0)
        plt.axvline(x =                  0, color = 'r', ymax = y_max, lw = 1.0)
        plt.axvline(x =   y_max_c, color = 'cyan', ymax = y_max, lw = 1.0)

        #plt.legend()
        plt.savefig(os.path.join(output_folder, "plane_fit_breakdown_c" + str(step) + ".png"), bbox_inches='tight')
        if abs(y_max_c) > breakdown_thres_c:
            if not test_run:
                plt.show()

        param_instance.increment()

    if test_run:
        print("plane_fit_breakdown OK")

if __name__ == "__main__":
    main(False) # test_run
