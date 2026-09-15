import sys
sys.path.append("../../pypi_package/src")

import pytest
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from gnc_smoothie.sup_gauss_newton import SupGaussNewton
from gnc_smoothie.gnc_null_params import GNC_NullParams
from gnc_smoothie.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie.welsch_influence_func import WelschInfluenceFunc
from gnc_smoothie.quadratic_influence_func import QuadraticInfluenceFunc

sys.path.append("../../pypi_package/tests")
from check_derivs import check_model_derivs, check_derivs

sys.path.append("../src/registration_3d")
from point_registration import PointRegistration

def randomM11() -> float:
    return 2.0*(np.random.rand() - 0.5)

def test_model_derivs():
    for test_idx in range(10):
        # ground-truth model
        model = np.array([randomM11(), randomM11(), randomM11(), randomM11(), randomM11(), randomM11()])

        # build data
        n_points = np.random.randint(5, 10)
        data = np.zeros((n_points,2,3))
        for i in range(n_points):
            for j in range(3):
                data[i][0][j] = randomM11()
                data[i][1][j] = randomM11()

        # check Python model
        model_instance = PointRegistration()
        assert(check_model_derivs(model_instance, model, data, include_2nd_derivs=True, diff_threshold_2nd_deriv=1.e-5))

        weight = np.zeros(n_points)
        for i in range(n_points):
            weight[i] = 0.2+np.random.rand()

        assert(check_derivs(SupGaussNewton(GNC_NullParams(QuadraticInfluenceFunc()), model_instance=model_instance),
                            model, data, weight=weight, diff_threshold_AlB=1.e-3))#, print_diffs=True, print_derivs=True))
        
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
        n_outliers = int(0.8*np.random.rand()*n_good_3d_point_pairs)

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

        param_instance = GNC_WelschParams(WelschInfluenceFunc(), 0.01, sigma_limit=50.0, num_sigma_steps=20)
        optimiser_instance = SupGaussNewton(param_instance, model_instance=PointRegistration())
        assert(optimiser_instance.fit(data))
        model = optimiser_instance.final_model
        R = optimiser_instance.final_model_ref
        for i in range(3):
            for j in range(3):
                assert(R[i][j] == pytest.approx(R_gt[i][j]))

        for i in range(3):
            assert(model[3+i] == pytest.approx(t_gt[i]))

if __name__ == "__main__":
    np.random.seed(43289) # We want the random numbers to be the same on each run
    test_model_derivs()
    test_answer()
