import math
import numpy as np
import os
import sys

if __name__ == "__main__":
    sys.path.append("../../../pypi_package/src")

from gnc_smoothie.sup_gauss_newton import SupGaussNewton
from gnc_smoothie.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie.linear_model.linear_regressor import LinearRegressor
from gnc_smoothie.cython_files.linear_regressor_welsch_evaluator import LinearRegressorWelschEvaluator

sys.path.append("../misc")
from minimiser import minimiser

def randomM11() -> float:
    return 2.0*(np.random.rand()-0.5)

def check_breakdown():
    n_points_xy = 20
    n_points = n_points_xy*n_points_xy
    outlier_ratio_list = np.linspace(0.01, 0.50, num=51)
    sigma = 0.1
    Dx = Dy = 2.0
    for outlier_ratio in outlier_ratio_list:
        data = np.zeros((n_points,3))

        bad_xy_av = np.zeros(2)
        n_bad_points = 0
        outlier_area = outlier_ratio*4.0/math.pi
        for i in range(n_points_xy):
            y = -Dy + i*2.0*Dy/(n_points_xy-1)
            for j in range(n_points_xy):
                x = -Dx + j*2.0*Dx/(n_points_xy-1)
                idx = i*n_points_xy+j
                data[idx][0] = x
                data[idx][1] = y
                xy_norm = [0.5*(x+Dx)/Dx, 0.5*(y+Dy)/Dy] # in range [0,1]
                #print("xy=",x,y,"xy_norm=",xy_norm)
                if xy_norm[0] ** 2 + xy_norm[1] ** 2 < outlier_area:
                    data[idx][2] = 2.0*sigma
                    bad_xy_av += [x,y]
                    n_bad_points += 1
                else:
                    data[idx][2] = 0.0

        bad_xy_av /= n_bad_points

        param_instance = GNC_WelschParams(WelschInfluenceFunc(), sigma_base=sigma)
        evaluator_instance = LinearRegressorWelschEvaluator(data[0])
        optimiser_instance = SupGaussNewton(param_instance, data, evaluator_instance=evaluator_instance)
        def objective_func(x: np.array) -> float:
            # The plane should intersect the point x = bad_xy_av[0], y = bad_xy_av[1]
            # So given a,b in z = a*x + b*y + c, we have c = z - a*x - b*y
            a = x[0]
            b = x[1]
            z = 2.0*sigma
            c = z - a*bad_xy_av[0] - b*bad_xy_av[1]
            return -optimiser_instance.objective_func(np.array([a,b,c]))

        ab_max,best_val = minimiser(objective_func, initial_centre=[0.0,0.0], initial_half_range=[2.0,2.0], n_samples=[41,41], scale_factor=1.4)
        good_val = optimiser_instance.objective_func(np.array([0.0,0.0,0.0]))
        print("Compare (",outlier_ratio,n_bad_points/n_points,")",-best_val,good_val,-best_val-good_val)
        
def main(test_run:bool, output_folder:str="../../../output", quick_run:bool=False):
    np.random.seed(0) # We want the numbers to be the same on each run

    check_breakdown()

    outlier_ratio = 0.26
    n_points_xy = 4 if quick_run else 10
    n_points = n_points_xy*n_points_xy
    n_bad_points = int(outlier_ratio*n_points)
    #print("n_bad_points=",n_bad_points)
    sigma_pop = 0.3
    q = 0.666667
    sigma_base = sigma_pop/q
    sigma_limit = 5.0
    num_sigma_steps = 3 if quick_run else 20

    plane_good = [0.0, 0.0, 0.0]

    Dx = Dy = 2.0
    data = np.zeros((n_points,3))
    for i in range(n_points_xy):
        y = -Dy + i*2.0*Dy/(n_points_xy-1)
        for j in range(n_points_xy):
            x = -Dx + j*2.0*Dx/(n_points_xy-1)
            idx = i*n_points_xy+j
            data[idx][0] = x
            data[idx][1] = y
            data[idx][2] = 0.0 #np.random.normal(0.0,sigma_pop) # good plane is a=b=0

    influence_func = WelschInfluenceFunc()
    model_instance = LinearRegressor(data[0])
    param_instance = GNC_WelschParams(influence_func, sigma_base,
                                      sigma_limit=sigma_limit, num_sigma_steps=num_sigma_steps)

    bad_point_scale = 5.0
    for test_idx in range(10000):
        data_c = np.copy(data)
        bad_point = 0.002*test_idx #randomM11()*bad_point_scale
        bad_point_count = 0
        for i in range(n_points_xy):
            for j in range(n_points_xy):
                idx = i*n_points_xy+j
                data_c[idx][2] = bad_point
                bad_point_count += 1
                if bad_point_count >= n_bad_points:
                    break

            if bad_point_count >= n_bad_points:
                break

        # calculate curvature at ground truth
        optimiser_instance = SupGaussNewton(param_instance, data_c, model_instance=model_instance)
        a, A = optimiser_instance.weighted_derivs(plane_good, 1.0) # lambda_b
        Asum = np.zeros((3,3))
        inv_sigma4 = math.pow(sigma_base, -4.0)
        #print("test_idx=",test_idx)
        for d in data_c:
            r = d[2] - plane_good[0]*d[0] - plane_good[1]*d[1] - plane_good[2] # residual
            #print("r=",r)
            H = np.array([d[0], d[1], 1.0])
            HTH = np.outer(H,H)
            Asum += inv_sigma4*math.exp(-0.5*r*r/(sigma_base*sigma_base))*(r*r - sigma_base*sigma_base)*HTH

        detA = np.linalg.det(Asum)
        #print("A=",A,"Asum=",Asum)
        if detA >= 0.0:
            print("A=",A,"Asum=",Asum)
            #print("data_c=",data_c)
            print("bad_point=",bad_point,"det=",detA)

        assert(detA < 0.0)

    if test_run:
        print("plane_fit_breakdown2 OK")

if __name__ == "__main__":
    main(False) # test_run
