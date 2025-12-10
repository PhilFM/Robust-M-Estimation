import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    import sys
    sys.path.append("../../pypi_package/src")

from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.gnc_null_params import GNC_NullParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc

from line_fit import LineFit

def lineFitFunc(a, b, optimiser_instance):
    return optimiser_instance.objective_func([a,b])

def main(test_run:bool, output_folder:str="../../Output"):
    # data is a list of [weight, value] pairs
    data = np.array([[-1.0, 0.0], [-0.5, 0.0], [0.0, 0.0], [0.5, 0.0], [1.0, 0.0], # good data
                     [-1.0, 0.5]]) # bad data
    weight = np.array([0.2, 0.2, 0.2, 0.2, 0.2, # good data
                       1.0]) # bad data

    alist = np.linspace(-5, 5, num=200)
    blist = np.linspace(-2.0, 2.0, num=200)
    param_instance = GNC_NullParams(WelschInfluenceFunc(0.2)) # sigma
    optimiser_instance = SupGaussNewton(param_instance, LineFit(), data, weight=weight)
    rmfv = np.vectorize(lineFitFunc, excluded={"optimiser_instance"})
    plt.close("all")
    plt.figure(num=1, dpi=120)
    plt.axline((-1.0,0),(1.0,0), color = 'b')
    plt.plot(blist, rmfv(alist, blist, optimiser_instance=optimiser_instance))

    #plt.legend()
    plt.savefig(os.path.join(output_folder, "line_fit_param_plot.png"), bbox_inches='tight')
    if not test_run:
        plt.show()

    if test_run:
        print("line_fit_param_plot OK")

if __name__ == "__main__":
    main(False) # test_run
