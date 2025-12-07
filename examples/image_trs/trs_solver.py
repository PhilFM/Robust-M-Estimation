import numpy as np

from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc

from trs import TRS

def main(testrun:bool, output_folder:str="../../Output"):
    np.random.seed(0) # We want the numbers to be the same on each run

    for test_idx in range(0,1):
        model_gt = [2.0*(np.random.rand()-0.5), 2.0*(np.random.rand()-0.5), 2.0*(np.random.rand()-0.5), 2.0*(np.random.rand()-0.5)]
        N = 6
        data = np.zeros((N*N,4))
        outlier_fraction = 0.5
        for i in range(N):
            for j in range(N):
                xyi = i*N+j
                data[xyi][0] = j
                data[xyi][1] = i
                if xyi < (1.0-outlier_fraction)*N*N:
                    data[xyi][2] = model_gt[1]*j - model_gt[0]*i + model_gt[2]
                    data[xyi][3] = model_gt[0]*j + model_gt[1]*i + model_gt[3]
                else:
                    # add outlier
                    data[xyi][2] = 10.0*2.0*(np.random.rand()-0.5)
                    data[xyi][3] = 10.0*2.0*(np.random.rand()-0.5)

        if not testrun:
            print("data=",data)

        influence_func_instance = WelschInfluenceFunc()
        param_instance = GNC_WelschParams(influence_func_instance, 0.2, 10.0, 50) # sigma_base, sigma_limit, num_sigma_steps
        sup_gn_instance = SupGaussNewton(param_instance, TRS(), data, max_niterations=100)
        if sup_gn_instance.run():
            model = sup_gn_instance.final_model

        if not testrun:
            print("model_gt=",model_gt,"model=",model)
            print("modelDiff=",model-model_gt)

    if testrun:
        print("trs_solver OK")

if __name__ == "__main__":
    main(False) # testrun
