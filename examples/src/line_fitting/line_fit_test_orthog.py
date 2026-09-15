import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.append("../../../pypi_package/src")
    sys.path.append("../../../pypi_package/src/gnc_smoothie/linear_model")
    sys.path.append("../../../pypi_package/src/gnc_smoothie/cython_files")

from line_fit_orthog_welsch import LineFitOrthogWelsch

def objective_func(a:float, b:float, optimiser_instance):
    return optimiser_instance.objective_func([a,b])

def gradient_func(a:float, b:float, optimiser_instance):
    a,AlB = optimiser_instance.weighted_derivs([a,b])
    return a

def randomM11() -> float:
    return 2.0*(np.random.rand()-0.5)

def test_with_sigma(line_gt, data_x, data_y, method:str, sigma:float, output_folder:str, test_run):

    # linear regression fitter y = a*x + b
    y_range = max(data_y) - min(data_y)

    # orthogonal regression fitter a*x + b*y + c = 0 where a^2+b^2=1
    line_fitter = LineFitOrthogWelsch(sigma, sigma_limit=y_range, debug=True)
    data_xp = np.reshape(data_x, (len(data_x),1))
    data_yp = np.reshape(data_y, (len(data_y),1))
    if line_fitter.fit(np.concatenate((data_xp, data_yp), axis=1), method=method):
        final_line = line_fitter.final_line
        final_weight = line_fitter.final_weight
        debug_line_list = line_fitter.debug_line_list
        if method == "BestAngle":
            (ref_a,ref_b) = (-line_fitter.debug_ref_line[0]/line_fitter.debug_ref_line[1],-line_fitter.debug_ref_line[2]/line_fitter.debug_ref_line[1])

        if not test_run:
            print("Orthogonal regression result: a,b,c=", final_line)
            line = np.array([-final_line[0]/final_line[1], -final_line[2]/final_line[1]])
            print("   error: ", line-line_gt)

        # change to True if you want to see the progress of the algorithm
        if False:
            for line in debug_line_list:
                if not test_run:
                    print(line)

    # get min and max of data
    x_min = min(data_x)
    x_max = max(data_x)
    y_min = min(data_y)
    y_max = max(data_y)

    # allow border
    border_size = 0.2
    xrange = x_max-x_min
    x_min -= border_size*xrange
    x_max += border_size*xrange
    yrange = y_max-y_min
    y_min -= border_size*yrange
    y_max += border_size*yrange

    plt.close("all")
    plt.figure(num=1, dpi=120)

    first_intermediate = True
    for line in debug_line_list:
        (a,b) = (-line[1][0]/line[1][1], -line[1][2]/line[1][1])
        color = (0.5+0.5*line[0], 1.0, 0.5+0.5*line[0])
        plt.axline((x_min, a*x_min+b), (x_max, a*x_max+b), color = color, linewidth=0.5, label = "Intermediate lines" if first_intermediate else None)
        first_intermediate = False

    ax = plt.gca()
    ax.set_aspect(1)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    plt.plot(data_x[0], data_y[0], color = (1,0,0), marker='o', label="Inlier data values") # will be overwritten with corrected colour
    plt.plot(data_x[0], data_y[0], color = (0,0,1), marker='o', label="Outlier data values") # will be overwritten with corrected colour
    max_weight = max(final_weight)
    for x,y,w in zip(data_x,data_y,final_weight, strict=True):
        alpha = w/max_weight
        color = [alpha, 0.0, 1.0-alpha]
        plt.plot(x, y, color = color, marker = 'o', label = "Intermediate lines" if first_intermediate else None)

    if method == "BestAngle":
        plt.axline((x_min, ref_a*x_min+ref_b), (x_max, ref_a*x_max+ref_b), color = "cyan", linewidth=1.0, label="Reference angle")

    (a,b) = (-final_line[0]/final_line[1], -final_line[2]/final_line[1])
    plt.axline((x_min, a*x_min+b), (x_max, a*x_max+b), color = "green", linewidth=1.5, label="Best fit line ("+method+")")

    plt.legend()
    plt.savefig(os.path.join(output_folder, "test_orthog_" + method.lower() + ".png"), bbox_inches='tight')
    if not test_run:
        plt.show()
        
def main(test_run:bool, output_folder:str="../../../output"):
    output_folder += "/line_fit"
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    np.random.seed(0) # We want the numbers to be the same on each run

    # data is a list of [x,y] pairs
    line_gt = [1.0, 0.0] # a,b
    n_good_points = 10
    n_bad_points = 1
    sigma_pop = 0.1

    # let's use the SciPy data format convention here, separating the "training data" X and "output" y
    data_x = np.zeros(n_good_points+n_bad_points)
    data_y = np.zeros(n_good_points+n_bad_points)
    for i in range(n_good_points):
        data_x[i] = -50 + 100.0*i/(n_good_points-1)
        data_y[i] = line_gt[0]*data_x[i] + line_gt[1] + np.random.normal(0.0, sigma_pop)

    (centre_x,centre_y) = (0.5*(data_x[0] + data_x[n_good_points-1]), 0.5*(data_y[0] + data_y[n_good_points-1]))
    if not test_run:
        print("centre xy:",centre_x,centre_y)

    for i in range(n_good_points,n_good_points+n_bad_points):
        centre_offset = 200.0
        data_x[i] = centre_x - centre_offset*line_gt[0]
        data_y[i] = centre_y + centre_offset

    p = 0.6667
    sigma_base = sigma_pop/p
    test_with_sigma(line_gt, data_x, data_y, "IRLS", sigma_base, output_folder, test_run)
    test_with_sigma(line_gt, data_x, data_y, "BestAngle", sigma_base, output_folder, test_run)

    if test_run:
        print("line_fit_test_orthog OK")

if __name__ == "__main__":
    main(False) # test_run
