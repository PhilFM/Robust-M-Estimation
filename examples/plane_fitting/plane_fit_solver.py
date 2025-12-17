import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    import sys
    sys.path.append("../../pypi_package/src")

from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc

from plane_fit_welsch        import PlaneFitWelsch
from plane_fit_orthog_welsch import PlaneFitOrthogWelsch

def objective_func(a:float, b:float, optimiser_instance):
    return optimiser_instance.objective_func([a,b])

def gradient_func(a:float, b:float, optimiser_instance):
    a,AlB = optimiser_instance.weighted_derivs([a,b])
    return a

def randomM11() -> float:
    return 2.0*(np.random.rand()-0.5)

def main(test_run:bool, output_folder:str="../../output"):
    np.random.seed(0) # We want the numbers to be the same on each run

    # data is a list of [x,y,z] triplets
    plane_gt = [0.5, 0.2, -1.0]
    n_good = 10
    n_bad = 0 #8
    data = np.zeros((n_good+n_bad,3))
    for i in range(n_good):
        data[i][0] = randomM11()
        data[i][1] = randomM11()
        data[i][2] = plane_gt[0]*data[i][0] + plane_gt[1]*data[i][1] + plane_gt[2]

    for i in range(n_bad):
        data[n_good+i][0] = randomM11()
        data[n_good+i][1] = randomM11()
        data[n_good+i][2] = randomM11()

    # linear regression fitter z = a*x + b*y + c
    plane_fitter = PlaneFitWelsch(0.01, 50.0, 20, debug=True)
    if plane_fitter.run(data):
        final_plane = plane_fitter.final_plane
        debug_plane_list = plane_fitter.debug_plane_list

    if not test_run:
        print("Linear regression result: a,b,c,d=", final_plane)
        plane = np.array([-final_plane[0]/final_plane[2], -final_plane[1]/final_plane[2], -final_plane[3]/final_plane[2]])
        print("   error: ", plane-plane_gt)

    # change to True if you want to see the progress of the algorithm
    if False:
        for plane in debug_plane_list:
            if not test_run:
                print(plane)

    # orthogonal regression fitter a*x + b*y + c*z + d = 0 where a^2+b^2+c^2=1
    plane_fitter_orthog = PlaneFitOrthogWelsch(0.01, 50.0, 20, debug=True)
    if plane_fitter_orthog.run(data):
        final_plane_orthog = plane_fitter_orthog.final_plane
        debug_plane_list_orthog = plane_fitter_orthog.debug_plane_list

    if not test_run:
        print("Orthogonal regression result: a,b,c,d=", final_plane_orthog)
        plane_orthog = np.array([-final_plane_orthog[0]/final_plane_orthog[2], -final_plane_orthog[1]/final_plane_orthog[2], -final_plane_orthog[3]/final_plane_orthog[2]])
        print("   error: ", plane_orthog-plane_gt)

    # change to True if you want to see the progress of the algorithm
    if False:
        for plane in debug_plane_list_orthog:
            if not test_run:
                print(plane)

    if test_run:
        print("plane_fit_solver OK")

if __name__ == "__main__":
    main(False) # test_run
