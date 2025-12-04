import numpy as np

from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc

from trs import TRS

def main(testrun:bool, output_folder:str="../../Output"):
    np.random.seed(0) # We want the numbers to be the same on each run

    for test_idx in range(0,1):
        modelGT = [2.0*(np.random.rand()-0.5), 2.0*(np.random.rand()-0.5), 2.0*(np.random.rand()-0.5), 2.0*(np.random.rand()-0.5)]
        N = 6
        data = np.zeros((N*N,4))
        outlier_fraction = 0.5
        for i in range(N):
            for j in range(N):
                xyi = i*N+j
                data[xyi][0] = j
                data[xyi][1] = i
                if xyi < (1.0-outlier_fraction)*N*N:
                    data[xyi][2] = modelGT[1]*j - modelGT[0]*i + modelGT[2]
                    data[xyi][3] = modelGT[0]*j + modelGT[1]*i + modelGT[3]
                else:
                    # add outlier
                    data[xyi][2] = 10.0*2.0*(np.random.rand()-0.5)
                    data[xyi][3] = 10.0*2.0*(np.random.rand()-0.5)

        if not testrun:
            print("data=",data)

        influence_func_instance = WelschInfluenceFunc()
        param_instance = GNC_WelschParams(influence_func_instance, 0.2, 10.0, 50) # sigma_base, sigma_limit, num_sigma_steps
        model = SupGaussNewton(param_instance, TRS(), data, max_niterations=100).run()
        if not testrun:
            print("modelGT=",modelGT,"model=",model)
            print("modelDiff=",model-modelGT)

    if testrun:
        print("trs_solver OK")

if __name__ == "__main__":
    main(False) # testrun
