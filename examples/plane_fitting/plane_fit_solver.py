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

def objective_func(a:float, b:float, optimiser_instance):
    return optimiser_instance.objective_func([a,b])

def gradient_func(a:float, b:float, optimiser_instance):
    a,AlB = optimiser_instance.weighted_derivs([a,b])
    return a

def main(test_run:bool, output_folder:str="../../output"):
    np.random.seed(0) # We want the numbers to be the same on each run

    # data is a list of [x,y,z] triplets
    plane_gt = [0.5, 0.2, -1.0]
    n_good = 10
    n_bad = 8
    data = np.zeros((n_good+n_bad,3))
    for i in range(n_good):
        data[i][0] = 2.0*(np.random.rand()-0.5)
        data[i][1] = 2.0*(np.random.rand()-0.5)
        data[i][2] = plane_gt[0]*data[i][0] + plane_gt[1]*data[i][1] + plane_gt[2]

    for i in range(n_bad):
        data[n_good+i][0] = 2.0*(np.random.rand()-0.5)
        data[n_good+i][1] = 2.0*(np.random.rand()-0.5)
        data[n_good+i][2] = 2.0*(np.random.rand()-0.5)

    param_instance = GNC_WelschParams(WelschInfluenceFunc(), 0.01, 50.0, 20) # sigma_base, sigma_limit, num_sigma_steps
    optimiser_instance = SupGaussNewton(param_instance, PlaneFit(), data, debug=True)
    if optimiser_instance.run():
        model = optimiser_instance.final_model
        debug_planes = optimiser_instance.debug_model_list

    if not test_run:
        print("Result: a,b,c=", model)

    # change to True if you want to see the progress of the algorithm
    if False:
        for plane in debug_planes:
            if not test_run:
                print(plane)

    if test_run:
        print("plane_fit_solver OK")

if __name__ == "__main__":
    main(False) # test_run
