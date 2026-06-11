import numpy as np
import matplotlib.pyplot as plt
import os
import math

if __name__ == "__main__":
    import sys
    sys.path.append("../../../pypi_package/src")
    sys.path.append("../../../pypi_package/src/gnc_smoothie/linear_model")
    sys.path.append("../../../pypi_package/src/gnc_smoothie/cython_files")

from gnc_smoothie.sup_gauss_newton import SupGaussNewton
from gnc_smoothie.linear_model.linear_regressor_welsch import LinearRegressorWelsch
from gnc_smoothie.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie.cython_files.linear_regressor_welsch_evaluator import LinearRegressorWelschEvaluator
from line_fit_orthog_welsch import LineFitOrthogWelsch

from gnc_smoothie.linear_model.linear_regressor_pseudo_huber import LinearRegressorPseudoHuber

sys.path.append("../misc")
from minimiser import minimiser

def objective_func(a:float, b:float, optimiser_instance):
    return optimiser_instance.objective_func([a,b])

def gradient_func(a:float, b:float, optimiser_instance):
    a,AlB = optimiser_instance.weighted_derivs([a,b])
    return a

def randomM11() -> float:
    return 2.0*(np.random.rand()-0.5)

def fit_line(data_x, data_y, sigma):
    y_range = max(data_y) - min(data_y)
    line_fitter = LinearRegressorWelsch(sigma, sigma_limit=y_range, num_sigma_steps=10, debug=True, max_niterations=200)
    assert(line_fitter.run((data_x, data_y)))
    coeff = line_fitter.final_coeff
    intercept = line_fitter.final_intercept
    return np.array([coeff[0][0], intercept[0]])

def check_gnc_variation(param_instance, data_x: np.array, data_y: np.array) -> None:
    small_diff = 0.001
    param_instance.reset()
    while True:
        sigma = math.sqrt(param_instance.influence_func_instance.variance())
        l1 = fit_line(data_x, data_y, sigma*(1.0 + small_diff))
        l2 = fit_line(data_x, data_y, sigma*(1.0 - small_diff))
        deriv = 0.5*(l2 - l1)/small_diff
        x_range = max(data_x) - min(data_x)
        print("sigma=",sigma,"deriv=",deriv,"sderiv=",x_range*deriv[0]/sigma,deriv[1]/sigma)
        if param_instance.alpha() == 1.0:
            break;

        param_instance.increment()

# get global maximum by sampling
def test_minimum(line_gt, data_x, data_y, sigma_base):
    data_xp = np.reshape(data_x, (len(data_x),1,1))
    data_yp = np.reshape(data_y, (len(data_y),1,1))
    data = np.concatenate((data_xp, data_yp), axis=2)
    sigma = 0.1 #sigma_base
    ab_max_list = []
    last_ab_max = None
    for i in range(50):
        param_instance = GNC_WelschParams(WelschInfluenceFunc(), sigma_base=sigma, sigma_limit=sigma)
        evaluator_instance = LinearRegressorWelschEvaluator(data[0])
        optimiser_instance = SupGaussNewton(param_instance, data, evaluator_instance = evaluator_instance)
        def objective_func(x: np.array) -> float:
            #print("x=",x)
            return -optimiser_instance.objective_func(x)

        #print("")
        ab_max,best_val = minimiser(objective_func, initial_centre=line_gt, initial_n_samples=[41,201], initial_half_range=[2.0, 50.0], n_samples=[21,21], scale_factor=2.0)
        ab_max_list.append(ab_max)
        if last_ab_max is not None:
            print("sigma=",sigma,"ab_max",ab_max,"ab diff:", ab_max[0]-last_ab_max[0], ab_max[1]-last_ab_max[1])

        last_ab_max = ab_max
        sigma *= 1.1

    return ab_max_list

def test_with_sigma(line_gt, data_x, data_y, sigma: float, output_folder: str, test_run: bool, ab_max_list):

    # linear regression fitter y = a*x + b
    x_range = max(data_x) - min(data_x)
    y_range = max(data_y) - min(data_y)
    line_fitter = LinearRegressorWelsch(sigma, sigma_limit=y_range, num_sigma_steps=50, debug=True, max_niterations=200,
                                        model_size_est=np.array([1.0/x_range, 1.0])) #, messages_file=sys.stdout)
    if line_fitter.run((data_x, data_y)):
        coeff = line_fitter.final_coeff
        intercept = line_fitter.final_intercept
        final_line = np.array([coeff[0][0], intercept[0]])
        final_weight = line_fitter.final_weight
        debug_line_list = line_fitter.debug_model_list
        check_gnc_variation(line_fitter.param_instance(), data_x, data_y)

    if not test_run:
        print("Linear regression result: a,b,c", final_line)
        print("   error: ", final_line-line_gt)

    # get min and max of data
    x_min = min(data_x)
    x_max = max(data_x)

    # allow border
    xrange = x_max-x_min
    x_min -= 0.05*xrange
    x_max += 0.05*xrange
    plt.close("all")
    plt.figure(num=1, dpi=120)

    cnt = 0
    print("ab_max_list=",ab_max_list)
    for ab_max in ab_max_list:
        color = (0.5, 0.5, cnt/(len(ab_max_list)-1))
        print("color=",color)
        plt.axline((x_min, ab_max[0]*x_min+ab_max[1]), (x_max, ab_max[0]*x_max+ab_max[1]), color = color, linewidth=0.5)
        cnt += 1

    plt.plot(data_x[0], data_y[0], color = (1,0,0), marker='o', label="Inlier data values") # will be overwritten with corrected colour
    plt.plot(data_x[0], data_y[0], color = (0,0,1), marker='o', label="Outlier data values") # will be overwritten with corrected colour
    max_weight = max(final_weight)
    for x,y,w in zip(data_x,data_y,final_weight, strict=True):
        alpha = w/max_weight
        color = [alpha, 0.0, 1.0-alpha]
        plt.plot(x, y, color = color, marker = 'o')

    #(a,b) = (final_line[0],final_line[1])
    #plt.axline((x_min, a*x_min+b), (x_max, a*x_max+b), color = "green", linewidth=1.5)

    plt.legend()
    plt.savefig(os.path.join(output_folder, "line_fit_solver.png"), bbox_inches='tight')
    if not test_run:
        plt.show()

    # show orthogonal regression result
    plt.close("all")
    plt.figure(num=1, dpi=120)
        
def main(test_run:bool, output_folder:str="../../../output"):
    np.random.seed(0) # We want the numbers to be the same on each run

    # data is a list of [x,y] pairs
    line_gt = [0.0, 0.0] # a,b
    n_good_points = 10
    n_bad_points = 4
    sigma_pop = 0.1

    # let's use the SciPy data format convention here, separating the "training data" X and "output" y
    data_x = np.zeros(n_good_points+n_bad_points)
    data_y = np.zeros(n_good_points+n_bad_points)
    for i in range(n_good_points):
        data_x[i] = -50 + 100.0*i/(n_good_points-1)
        data_y[i] = line_gt[0]*data_x[i] + line_gt[1] # + np.random.normal(0.0, sigma_pop)

    for i in range(n_good_points,n_good_points+n_bad_points):
        data_x[i] = -50 # + 1.0+1.5*(i-n_good_points)
        while(True):
            data_y[i] = -8 # -15 + 15.0*np.random.rand() #line_gt[0]*data[i][0] + line_gt[1] + 0.2 + 0.1*np.random.rand()
            residual = line_gt[0]*data_x[i] + line_gt[1] - data_y[i]
            if abs(residual) > 5.0*sigma_pop:
                break

    # with small error estimate we will fit to the good data only
    p = 0.6667
    sigma_base = sigma_pop/p
    ab_max_list = test_minimum(line_gt, data_x, data_y, sigma_base*10.0)
    test_with_sigma(line_gt, data_x, data_y, sigma_base, output_folder, test_run, ab_max_list)

    # with a larger error estimate the points close to the good data will influence the result
    #test_with_sigma(line_gt, data_x, data_y, 2.0, output_folder, test_run)

    if test_run:
        print("line_fit_solver OK")

if __name__ == "__main__":
    main(False) # test_run
