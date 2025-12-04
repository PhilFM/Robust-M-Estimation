import numpy as np
from scipy.spatial.transform import Rotation as Rot

from gnc_smoothie_philfm.null_params import NullParams
from gnc_smoothie_philfm.quadratic_influence_func import QuadraticInfluenceFunc
from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.check_derivs import check_derivs

from point_registration import PointRegistration

def main(testrun:bool, output_folder:str="../../Output"):
    np.random.seed(0) # We want the numbers to be the same on each run

    N = 10
    noise_sigma = 0.5

    for test_idx in range(0,1):
        t = np.zeros(3)
        t[0] = np.random.normal(0.0, 1.0)
        t[1] = np.random.normal(0.0, 1.0)
        t[2] = np.random.normal(0.0, 1.0)

        R = Rot.random().as_matrix()
        if not testrun:
            print("R=",R,"t=",t)

        data = np.zeros((N,2,3))
        weight = np.zeros(N)
        for i in range(0,N):
            data[i][0][0] = np.random.normal(0.0, 1.0)
            data[i][0][1] = np.random.normal(0.0, 1.0)
            data[i][0][2] = np.random.normal(0.0, 1.0)
            RX = np.matmul(R,data[i][0])
            RXpt = RX + t
            data[i][1] = RXpt
            data[i][1][0] += noise_sigma*np.random.normal(0.0, 1.0)
            data[i][1][1] += noise_sigma*np.random.normal(0.0, 1.0)
            data[i][1][2] += noise_sigma*np.random.normal(0.0, 1.0)
            weight[i] = 1.0

        all_good = True
        if check_derivs(SupGaussNewton(NullParams(QuadraticInfluenceFunc()), PointRegistration(), data, weight=weight),
                        np.array([0.0,0.0,0.0,t[0],t[1],t[2]]), model_ref=R, diff_threshold_AlB=1.e-4, print_derivs=False, print_diffs=False) is False:
            all_good = False

    if all_good:
        if testrun:
            print("registration_deriv_check OK")
        else:
            print("ALL DERIVATIVES OK!!")
    else:
        print("Derivative failure")

if __name__ == "__main__":
    main(False) # testrun
