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

def test_with_sigma(line_gt, data_x, data_y, sigma: float, output_folder: str, test_run):

    # linear regression fitter y = a*x + b
    x_range = max(data_x) - min(data_x)
    y_range = max(data_y) - min(data_y)
    line_fitter = LinearRegressorWelsch(sigma, sigma_limit=y_range, num_sigma_steps=10, debug=True, max_niterations=200)
    #model_size_est=np.array([1.0/x_range, 1.0])) #, messages_file=sys.stdout)
    if line_fitter.run((data_x, data_y)):
        coeff = line_fitter.final_coeff
        intercept = line_fitter.final_intercept
        final_line = np.array([coeff[0][0], intercept[0]])
        final_weight = line_fitter.final_weight
        debug_line_list = line_fitter.debug_model_list

    if not test_run:
        print("Linear regression result: a,b,c", final_line)
        print("   error: ", final_line-line_gt)

    test_orthogonal_regression = False
    if test_orthogonal_regression:
        # orthogonal regression fitter a*x + b*y + c = 0 where a^2+b^2=1
        line_fitter_orthog = LineFitOrthogWelsch(sigma, y_range, 20, debug=True)
        if line_fitter_orthog.run(np.concatenate((np.reshape(data_x, (len(data_x),1)),
                                                  np.reshape(data_y, (len(data_y),1))), axis=1)):
            final_line_orthog = line_fitter_orthog.final_line
            final_weight_orthog = line_fitter_orthog.final_weight
            debug_line_list_orthog = line_fitter_orthog.debug_line_list

        if not test_run:
            print("Orthogonal regression result: a,b,c=", final_line_orthog)
            line_orthog = np.array([-final_line_orthog[0]/final_line_orthog[1], -final_line_orthog[2]/final_line_orthog[1]])
            print("   error: ", line_orthog-line_gt)

        # change to True if you want to see the progress of the algorithm
        if False:
            for line in debug_line_list_orthog:
                if not test_run:
                    print(line)

    # get min and max of data
    x_min = min(data_x)
    x_max = max(data_x)

    # allow border
    xrange = x_max-x_min
    x_min -= 0.05*xrange
    x_max += 0.05*xrange
    plt.close("all")
    plt.figure(num=1, dpi=120)

    # change to True if you want to see the progress of the algorithm
    if True:
        last_alpha = None
        max_alpha = debug_line_list[len(debug_line_list)-1][0]
        for line in debug_line_list:
            alpha = line[0]/max_alpha
            if last_alpha is None or alpha != last_alpha:
                coeff = line[1][0]
                intercept = line[1][1]
                (a,b) = (coeff[0][0], intercept[0])
                color = (0.1, alpha, 1.0-1.0*alpha)
                print("alpha=",line[0],"color=",color)
                plt.axline((x_min, a*x_min+b), (x_max, a*x_max+b), color = color, linewidth=0.5)
                last_alpha = alpha

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

    # change to True if you want to see the progress of the algorithm
    if test_orthogonal_regression:
        for line in debug_line_list_orthog:
            (a,b) = (-line_orthog[1][0]/line_orthog[1][1], -line_orthog[1][2]/line_orthog[1][1])
            color = (0.5+0.5*line_orthog[0], 1.0, 0.5+0.5*line_orthog[0])
            plt.axline((x_min, a*x_min+b), (x_max, a*x_max+b), color = color, linewidth=0.5)

        plt.plot(data_x[0], data_y[0], color = (1,0,0), marker='o', label="Inlier data values") # will be overwritten with corrected colour
        plt.plot(data_x[0], data_y[0], color = (0,0,1), marker='o', label="Outlier data values") # will be overwritten with corrected colour
        max_weight = max(final_weight_orthog)
        for x,y,w in zip(data_x,data_y,final_weight_orthog, strict=True):
            alpha = w/max_weight
            color = [alpha, 0.0, 1.0-alpha]
            plt.plot(x, y, color = color, marker = 'o')

        (a,b) = (-final_line_orthog[0]/final_line_orthog[1], -final_line_orthog[2]/final_line_orthog[1])
        plt.axline((x_min, a*x_min+b), (x_max, a*x_max+b), color = "green", linewidth=1.5)

        plt.legend()
        plt.savefig(os.path.join(output_folder, "line_fit_solver_orthog.png"), bbox_inches='tight')
        if not test_run:
            plt.show()
        
def test_with_sigma_pseudo_huber(line_gt, data_x, data_y, sigma: float, output_folder: str, test_run: bool):
    # linear regression fitter y = a*x + b
    x_range = max(data_x) - min(data_x)
    y_range = max(data_y) - min(data_y)
    line_fitter = LinearRegressorPseudoHuber(sigma, debug=True, max_niterations=200,
                                             messages_file=sys.stdout)
    if line_fitter.run((data_x, data_y), model_start=[randomM11(),randomM11()]):
        coeff = line_fitter.final_coeff
        intercept = line_fitter.final_intercept
        final_line = np.array([coeff[0][0], intercept[0]])
        final_weight = line_fitter.final_weight
        debug_line_list = line_fitter.debug_model_list

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

    # change to True if you want to see the progress of the algorithm
    if True:
        last_alpha = None
        for line in debug_line_list:
            alpha = line[0]
            if last_alpha is None or alpha != last_alpha:
                coeff = line[1][0]
                intercept = line[1][1]
                (a,b) = (coeff[0][0], intercept[0])
                color = (0.1, alpha, 1.0-1.0*alpha)
                plt.axline((x_min, a*x_min+b), (x_max, a*x_max+b), color = color, linewidth=0.5)
                last_alpha = alpha

    plt.plot(data_x[0], data_y[0], color = (1,0,0), marker='o', label="Inlier data values") # will be overwritten with corrected colour
    plt.plot(data_x[0], data_y[0], color = (0,0,1), marker='o', label="Outlier data values") # will be overwritten with corrected colour
    max_weight = max(final_weight)
    for x,y,w in zip(data_x,data_y,final_weight, strict=True):
        alpha = w/max_weight
        color = [alpha, 0.0, 1.0-alpha]
        plt.plot(x, y, color = color, marker = 'o')

    (a,b) = (final_line[0],final_line[1])
    plt.axline((x_min, a*x_min+b), (x_max, a*x_max+b), color = "green", linewidth=1.5)

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
    line_gt = [0.5, 0.9] # a,b
    n_good_points = 10
    n_bad_points = 9
    sigma_pop = 0.1

    # let's use the SciPy data format convention here, separating the "training data" X and "output" y
    data_x = np.zeros(n_good_points+n_bad_points)
    data_y = np.zeros(n_good_points+n_bad_points)
    for i in range(n_good_points):
        data_x[i] = -50 + 100.0*i/(n_good_points-1)
        data_y[i] = line_gt[0]*data_x[i] + line_gt[1] + np.random.normal(0.0, sigma_pop)

    for i in range(n_good_points,n_good_points+n_bad_points):
        data_x[i] = -40 + 12*(i-n_good_points)
        while(True):
            data_y[i] = -30.0 + 60.0*np.random.rand() #line_gt[0]*data[i][0] + line_gt[1] + 0.2 + 0.1*np.random.rand()
            residual = line_gt[0]*data_x[i] + line_gt[1] - data_y[i]
            if abs(residual) > 5.0*sigma_pop:
                break

    # with small error estimate we will fit to the good data only
    p = 0.6667
    sigma_base = sigma_pop/p
    test_with_sigma(line_gt, data_x, data_y, sigma_base, output_folder, test_run)
    #for test_idx in range(10):
    #    test_with_sigma_pseudo_huber(line_gt, data_x, data_y, sigma_pop/p, output_folder, test_run)

    # with a larger error estimate the points close to the good data will influence the result
    #test_with_sigma(line_gt, data_x, data_y, 2.0, output_folder, test_run)

    if test_run:
        print("line_fit_solver OK")

if __name__ == "__main__":
    main(False) # test_run
