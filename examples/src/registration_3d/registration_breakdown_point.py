import numpy as np
from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    import sys
    sys.path.append("../../../pypi_package/src")

from gnc_smoothie.sup_gauss_newton import SupGaussNewton
from gnc_smoothie.irls import IRLS
from gnc_smoothie.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie.gnc_irls_p_params import GNC_IRLSpParams
from gnc_smoothie.gnc_null_params import GNC_NullParams
from gnc_smoothie.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie.pseudo_huber_influence_func import PseudoHuberInfluenceFunc
from gnc_smoothie.gnc_irls_p_influence_func import GNC_IRLSpInfluenceFunc
from gnc_smoothie.plt_alg_vis import gncs_draw_curve

sys.path.append("../misc")
from minimiser import minimiser

from point_registration import PointRegistration

def main(test_run:bool, output_folder:str="../../../output"):
    np.random.seed(0) # We want the numbers to be the same on each run
    n_xyz_samples = 3 # along each of the 3 dimensions
    n_good_samples = n_xyz_samples*n_xyz_samples*n_xyz_samples
    outlier_ratio = 0.0 #0.5

    # f(0) = 0, f(0.1) = 0.11111, f(0.5) = 1, f(1) = inf
    # or = x/(1+x), or + x*or = x, x*(1-or) = or, x = or/(1-or)
    n_bad_samples = int(0.5 + n_good_samples*outlier_ratio/(1.0 - outlier_ratio))
    N = n_good_samples + n_bad_samples
    sigma = 0.5 # noise
    sigma_limit = 10.0
    welsch_p = 0.666667
    sigma_base = sigma/welsch_p
    num_sigma_steps = 10
    translation_bound    = 10.0

    for test_idx in range(0,1):
        t_gt = np.zeros(3)
        t_gt[0] = np.random.normal(0.0, 1.0)
        t_gt[1] = np.random.normal(0.0, 1.0)
        t_gt[2] = np.random.normal(0.0, 1.0)
        t_gt /= np.linalg.norm(t_gt)
        t_gt = (translation_bound) * np.random.rand() * t_gt

        R_gt = Rot.random().as_matrix()
        if not test_run:
            print("Ground truth R=",R_gt,"t=",t_gt)

        data = np.zeros((N,2,3))
        weight = np.ones(N)

        xyz_range = 1.0
        sample = 0
        for i in range(0,n_xyz_samples):
            x = -0.5*xyz_range + xyz_range*i/(n_xyz_samples-1)
            for j in range(0,n_xyz_samples):
                y = -0.5*xyz_range + xyz_range*j/(n_xyz_samples-1)
                for k in range(0,n_xyz_samples):
                    z = -0.5*xyz_range + xyz_range*k/(n_xyz_samples-1)
                    data[sample][0][0] = x
                    data[sample][0][1] = y
                    data[sample][0][2] = z
                    #print("data[i][0]=",data[i][0])
                    RX = np.matmul(R_gt,data[sample][0])
                    #print("RX=",RX)
                    RXpt = RX + t_gt
                    data[i][1] = RXpt

        for i in range(n_good_samples,N):
            data[i][0][0] = xyz_range*(np.random.rand() - 0.5)
            data[i][0][1] = xyz_range*(np.random.rand() - 0.5)
            data[i][0][2] = xyz_range*(np.random.rand() - 0.5)
            data[i][1][0] = xyz_range*(np.random.rand() - 0.5)
            data[i][1][1] = xyz_range*(np.random.rand() - 0.5)
            data[i][1][2] = xyz_range*(np.random.rand() - 0.5)

        welsch_param_instance = GNC_WelschParams(WelschInfluenceFunc(),
                                                 sigma_base=sigma_base, sigma_limit=sigma_limit,
                                                 num_sigma_steps=num_sigma_steps)
        optimiser_instance = SupGaussNewton(welsch_param_instance, data, model_instance=PointRegistration())
        optimiser_instance._param_instance.reset(init=False)

        def objective_func(x: np.array) -> float:
            #print("x=",x)
            xp = np.copy(x)
            R = Rot.as_matrix(Rot.from_mrp([0.25*x[0],0.25*x[1],0.25*x[2]]))
            xp[0] = xp[1] = xp[2] = 0.0
            return -optimiser_instance.objective_func(xp, model_ref=R)

        #print("")
        rt_max,best_val = minimiser(objective_func, initial_centre=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], initial_half_range=[1.0, 1.0, 1.0, 5.0, 5.0, 5.0], n_samples=[7,7,7,7,7,7], scale_factor=1.5)

    if test_run:
        print("registration_breakdown_point OK")

if __name__ == "__main__":
    main(False) # test_run
