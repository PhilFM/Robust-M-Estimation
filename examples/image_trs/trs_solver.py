import numpy as np

if __name__ == "__main__":
    import sys
    sys.path.append("../../pypi_package/src")

from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc

from trs_welsch import TRSWelsch

def main(test_run:bool, output_folder:str="../../output"):
    np.random.seed(0) # We want the numbers to be the same on each run

    for test_idx in range(0,1):
        model_gt = [2.0*(np.random.rand()-0.5), 2.0*(np.random.rand()-0.5), 2.0*(np.random.rand()-0.5), 2.0*(np.random.rand()-0.5)]
        n = 10
        data = np.zeros((n*n,4))
        outlier_fraction = 0.5
        for i in range(n):
            for j in range(n):
                xyi = i*n+j
                data[xyi][0] = j
                data[xyi][1] = i
                if xyi < (1.0-outlier_fraction)*n*n:
                    data[xyi][2] = model_gt[1]*j - model_gt[0]*i + model_gt[2]
                    data[xyi][3] = model_gt[0]*j + model_gt[1]*i + model_gt[3]
                else:
                    # add outlier
                    data[xyi][2] = 10.0*2.0*(np.random.rand()-0.5)
                    data[xyi][3] = 10.0*2.0*(np.random.rand()-0.5)

        if not test_run:
            print("data=",data)

        trs = TRSWelsch(0.2, 10.0, 50, max_niterations=100, debug=True)
        if trs.run(data):
            model = trs.final_model

        if not test_run:
            print("model_gt=",model_gt,"model=",model)
            print("modelDiff=",model-model_gt)
            print("n_iterations:",trs.debug_n_iterations)

    if test_run:
        print("trs_solver OK")

if __name__ == "__main__":
    main(False) # test_run
