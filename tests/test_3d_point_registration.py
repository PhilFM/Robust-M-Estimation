import sys
sys.path.append("../pypi_package/src")

import pytest
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc

sys.path.append("../examples/registration_3d")
from point_registration import PointRegistration

def test_answer():
    np.random.seed(0) # We want the numbers to be the same on each run
    for test_idx in range(10):
        t_gt = np.zeros(3)
        t_gt[0] = np.random.normal(0.0, 1.0)
        t_gt[1] = np.random.normal(0.0, 1.0)
        t_gt[2] = np.random.normal(0.0, 1.0)
        t_gt /= np.linalg.norm(t_gt)
        t_gt = 6.0 * np.random.rand() * t_gt

        R_gt = Rot.random().as_matrix()

        # build good data
        n_good_3d_point_pairs = np.random.randint(5, 20)
        n_outliers = int(np.random.rand()*n_good_3d_point_pairs)

        data = np.zeros((n_good_3d_point_pairs+n_outliers,2,3))
        for i in range(n_good_3d_point_pairs):
            data[i][0][0] = 3*(np.random.rand() - 0.5)
            data[i][0][1] = 3*(np.random.rand() - 0.5)
            data[i][0][2] = 3*(np.random.rand() - 0.5)
            RXpt = np.matmul(R_gt,data[i][0]) + t_gt
            data[i][1] = RXpt

        for i in range(n_outliers):
            data[n_good_3d_point_pairs+i][0][0] = 3*(np.random.rand() - 0.5)
            data[n_good_3d_point_pairs+i][0][1] = 3*(np.random.rand() - 0.5)
            data[n_good_3d_point_pairs+i][0][2] = 3*(np.random.rand() - 0.5)
            data[n_good_3d_point_pairs+i][1][0] = 3*(np.random.rand() - 0.5)
            data[n_good_3d_point_pairs+i][1][1] = 3*(np.random.rand() - 0.5)
            data[n_good_3d_point_pairs+i][1][2] = 3*(np.random.rand() - 0.5)

        param_instance = GNC_WelschParams(WelschInfluenceFunc(), 0.01, 50.0, 20) # sigma_base, sigma_limit, num_sigma_steps
        optimiser_instance = SupGaussNewton(param_instance, PointRegistration(), data)
        assert(optimiser_instance.run())
        model = optimiser_instance.final_model
        R = optimiser_instance.final_model_ref
        print("R=",R)
        for i in range(3):
            for j in range(3):
                assert(R[i][j] == pytest.approx(R_gt[i][j]))

        for i in range(3):
            assert(model[3+i] == pytest.approx(t_gt[i]))

if __name__ == "__main__":
    test_answer()
