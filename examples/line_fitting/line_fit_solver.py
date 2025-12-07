import numpy as np
import matplotlib.pyplot as plt
import os

from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc

from line_fit import LineFit

def objective_func(a, b, optimiser_instance):
    return optimiser_instance.objective_func([a,b])

def gradient_func(a, b, optimiser_instance):
    a,AlB = optimiser_instance.weighted_derivs([a,b])
    return a

def main(testrun:bool, output_folder:str="../../Output"):
    # data is a list of [x,y] pairs
    data = np.array([[0.0, 0.90], [0.1, 0.95], [0.2, 1.0], [0.3, 1.05], [0.4, 1.1], # good data
                     [0.05, 10.0], [0.15, 2.0], [0.25, 2.5], [0.35, 3.2]]) # bad data

    sigma_base = 2
    param_instance = GNC_WelschParams(WelschInfluenceFunc(), sigma_base, 50.0, 20) # sigma_base, sigma_limit, num_sigma_steps
    optimiser_instance = SupGaussNewton(param_instance, LineFit(), data, debug=True)
    if optimiser_instance.run():
        model = optimiser_instance.final_model
        debug_lines = optimiser_instance.debug_model_list

    if not testrun:
        print("Result: a,b=", model)

    # get min and max of data
    xmin = xmax = data[0][0]
    for d in data:
        xmin = min(xmin, d[0])
        xmax = max(xmax, d[0])

    # allow border
    xrange = xmax-xmin
    xmin -= 0.05*xrange
    xmax += 0.05*xrange
    plt.close("all")
    plt.figure(num=1, dpi=120)

    # change to True if you want to see the progress of the algorithm
    if False:
        for line in debug_lines:
            if not testrun:
                print(line)

            plt.axline((xmin, line[1][0]*xmin+line[1][1]), (xmax, line[1][0]*xmax+line[1][1]), color = (1.0-line[0], line[0], 1.0), linewidth=0.5)

    for d in data:
        plt.plot(d[0], d[1], color = 'b', marker = 'o')

    plt.axline((xmin, model[0]*xmin+model[1]), (xmax, model[0]*xmax+model[1]), color = "green", linewidth=1.5)

    plt.savefig(os.path.join(output_folder, "line_fit_solver.png"), bbox_inches='tight')
    if not testrun:
        plt.show()

    if testrun:
        print("line_fit_solver OK")

if __name__ == "__main__":
    main(False) # testrun
