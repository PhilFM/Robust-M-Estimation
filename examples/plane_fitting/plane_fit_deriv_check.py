import numpy as np

if __name__ == "__main__":
    import sys
    sys.path.append("../../pypi_package/src")

from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.gnc_null_params import GNC_NullParams
from gnc_smoothie_philfm.quadratic_influence_func import QuadraticInfluenceFunc
from gnc_smoothie_philfm.check_derivs import check_derivs

from plane_fit import PlaneFit

def main(test_run:bool, output_folder:str="../../Output"):
    np.random.seed(0) # We want the numbers to be the same on each run

    all_good = True
    for test_idx in range(0,10):
        data = np.zeros((1,3))
        weight = np.zeros(1)
        weight[0] = 1.0
        for i in range(3):
            data[0][i] = 2.0*(np.random.rand()-0.5)

        # ground-truth parameters
        model = np.array([2.0*(np.random.rand()-0.5), # a
                          2.0*(np.random.rand()-0.5), # b
                          2.0*(np.random.rand()-0.5)]) # c

        optimiser_instance = SupGaussNewton(GNC_NullParams(QuadraticInfluenceFunc()), PlaneFit(), data, weight=weight)
        if not check_derivs(optimiser_instance, model, diff_threshold_AlB=1.e-5): #, print_diffs=True, print_derivs=True):
            all_good = False

    if all_good:
        if test_run:
            print("plane_fit_deriv_check OK")
        else:
            print("ALL DERIVATIVES OK!!")
    else:
        print("Derivative failure")

if __name__ == "__main__":
    main(False) # test_run
