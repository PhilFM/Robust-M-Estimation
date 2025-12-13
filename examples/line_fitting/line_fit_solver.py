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

def objective_func(a:float, b:float, optimiser_instance):
    return optimiser_instance.objective_func([a,b])

def gradient_func(a:float, b:float, optimiser_instance):
    a,AlB = optimiser_instance.weighted_derivs([a,b])
    return a

def test_with_sigma(sigma: float, output_folder: str, test_run: bool):
    # data is a list of [x,y] pairs
    data = np.array([[0.0, 0.90], [0.1, 0.95], [0.2, 1.0], [0.3, 1.05], [0.4, 1.1], # good data
                     [0.05, 10.0], [0.15, 2.0], [0.25, 2.5], [0.35, 3.2]]) # bad data

    param_instance = GNC_WelschParams(WelschInfluenceFunc(), sigma, 50.0, 20) # sigma_base, sigma_limit, num_sigma_steps
    optimiser_instance = SupGaussNewton(param_instance, LineFit(), data, debug=True)
    if optimiser_instance.run():
        model = optimiser_instance.final_model
        debug_lines = optimiser_instance.debug_model_list

    if not test_run:
        print("Result: a,b=", model)

    # get min and max of data
    x_min = x_max = data[0][0]
    for d in data:
        x_min = min(x_min, d[0])
        x_max = max(x_max, d[0])

    # allow border
    xrange = x_max-x_min
    x_min -= 0.05*xrange
    x_max += 0.05*xrange
    plt.close("all")
    plt.figure(num=1, dpi=120)

    # change to True if you want to see the progress of the algorithm
    if False:
        for line in debug_lines:
            if not test_run:
                print(line)

            plt.axline((x_min, line[1][0]*x_min+line[1][1]), (x_max, line[1][0]*x_max+line[1][1]), color = (1.0-line[0], line[0], 1.0), linewidth=0.5)

    for d in data:
        plt.plot(d[0], d[1], color = 'b', marker = 'o')

    plt.axline((x_min, model[0]*x_min+model[1]), (x_max, model[0]*x_max+model[1]), color = "green", linewidth=1.5)

    plt.savefig(os.path.join(output_folder, "line_fit_solver.png"), bbox_inches='tight')
    if not test_run:
        plt.show()

def main(test_run:bool, output_folder:str="../../output"):
    # with small error estimate we will fit to the good data only
    test_with_sigma(0.2, output_folder, test_run)

    # with a larger error estimate the points close to the good data will influence the result
    test_with_sigma(2.0, output_folder, test_run)

    if test_run:
        print("line_fit_solver OK")

if __name__ == "__main__":
    main(False) # test_run
