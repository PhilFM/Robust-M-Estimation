import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    import sys
    sys.path.append("../../pypi_package/src")

from line_fit_welsch        import LineFitWelsch
from line_fit_orthog_welsch import LineFitOrthogWelsch

def objective_func(a:float, b:float, optimiser_instance):
    return optimiser_instance.objective_func([a,b])

def gradient_func(a:float, b:float, optimiser_instance):
    a,AlB = optimiser_instance.weighted_derivs([a,b])
    return a

def randomM11() -> float:
    return 2.0*(np.random.rand()-0.5)

def test_with_sigma(line_gt, data, sigma: float, output_folder: str, test_run: bool):
    # linear regression fitter y = a*x + b
    y_range = max(data[:,1]) - min(data[:,1])
    line_fitter = LineFitWelsch(sigma, y_range, 20, debug=True)
    if line_fitter.run(data):
        final_line = line_fitter.final_line
        final_weight = line_fitter.final_weight
        debug_line_list = line_fitter.debug_line_list

    if not test_run:
        print("Linear regression result: a,b,c", final_line)
        line = np.array([-final_line[0]/final_line[1], -final_line[2]/final_line[1]])
        print("   error: ", line-line_gt)

    # orthogonal regression fitter a*x + b*y + c = 0 where a^2+b^2=1
    line_fitter_orthog = LineFitOrthogWelsch(sigma, y_range, 20, debug=True)
    if line_fitter_orthog.run(data):
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
    x_min = min(data[:,0])
    x_max = max(data[:,0])

    # allow border
    xrange = x_max-x_min
    x_min -= 0.05*xrange
    x_max += 0.05*xrange
    plt.close("all")
    plt.figure(num=1, dpi=120)

    # change to True if you want to see the progress of the algorithm
    if False:
        for line in debug_line_list:
            if not test_run:
                print(line)

            (a,b) = (-line[1][0]/line[1][1], -line[1][2]/line[1][1])
            plt.axline((x_min, a*x_min+b), (x_max, a*x_max+b), color = (1.0-line[0], line[0], 1.0), linewidth=0.5)

    plt.plot(data[0][0], data[0][1], color = (1,0,0), marker='o', label="Inlier data values") # will be overwritten with corrected colour
    plt.plot(data[0][0], data[0][1], color = (0,0,1), marker='o', label="Outlier data values") # will be overwritten with corrected colour
    max_weight = max(final_weight)
    for d,w in zip(data,final_weight, strict=True):
        alpha = w/max_weight
        color = [alpha, 0.0, 1.0-alpha]
        plt.plot(d[0], d[1], color = color, marker = 'o')

    (a,b) = (-final_line[0]/final_line[1], -final_line[2]/final_line[1])
    plt.axline((x_min, a*x_min+b), (x_max, a*x_max+b), color = "green", linewidth=1.5)

    plt.legend()
    plt.savefig(os.path.join(output_folder, "line_fit_solver.png"), bbox_inches='tight')
    if not test_run:
        plt.show()

    # show orthogonal regression result
    plt.close("all")
    plt.figure(num=1, dpi=120)

    # change to True if you want to see the progress of the algorithm
    if False:
        for line in debug_line_list_orthog:
            (a,b) = (-line_orthog[1][0]/line_orthog[1][1], -line_orthog[1][2]/line_orthog[1][1])
            plt.axline((x_min, a*x_min+b), (x_max, a*x_max+b), color = (1.0-line_orthog[0], line_orthog[0], 1.0), linewidth=0.5)

    plt.plot(data[0][0], data[0][1], color = (1,0,0), marker='o', label="Inlier data values") # will be overwritten with corrected colour
    plt.plot(data[0][0], data[0][1], color = (0,0,1), marker='o', label="Outlier data values") # will be overwritten with corrected colour
    max_weight = max(final_weight_orthog)
    for d,w in zip(data,final_weight_orthog, strict=True):
        alpha = w/max_weight
        color = [alpha, 0.0, 1.0-alpha]
        plt.plot(d[0], d[1], color = color, marker = 'o')

    (a,b) = (-final_line_orthog[0]/final_line_orthog[1], -final_line_orthog[2]/final_line_orthog[1])
    plt.axline((x_min, a*x_min+b), (x_max, a*x_max+b), color = "green", linewidth=1.5)

    plt.legend()
    plt.savefig(os.path.join(output_folder, "line_fit_solver_orthog.png"), bbox_inches='tight')
    if not test_run:
        plt.show()
        
def main(test_run:bool, output_folder:str="../../output"):
    np.random.seed(0) # We want the numbers to be the same on each run

    # data is a list of [x,y] pairs
    line_gt = [0.5, 0.9] # a,b
    n_good_points = 10
    n_bad_points = 8
    sigma_pop = 0.1
    data = np.zeros((n_good_points+n_bad_points,2))
    for i in range(n_good_points):
        data[i][0] = 1.0*i
        data[i][1] = line_gt[0]*data[i][0] + line_gt[1] + np.random.normal(0.0, sigma_pop)

    for i in range(n_good_points,n_good_points+n_bad_points):
        data[i][0] = 1.0+0.9*(i-n_good_points)
        while(True):
            data[i][1] = 7.0*np.random.rand() #line_gt[0]*data[i][0] + line_gt[1] + 0.2 + 0.1*np.random.rand()
            residual = line_gt[0]*data[i][0] + line_gt[1] - data[i][1]
            if abs(residual) > 5.0*sigma_pop:
                break
        
    # with small error estimate we will fit to the good data only
    p = 0.6667
    test_with_sigma(line_gt, data, sigma_pop/p, output_folder, test_run)

    # with a larger error estimate the points close to the good data will influence the result
    #test_with_sigma(line_gt, data, 2.0, output_folder, test_run)

    if test_run:
        print("line_fit_solver OK")

if __name__ == "__main__":
    main(False) # test_run
