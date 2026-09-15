import sys
sys.path.append("../../pypi_package/src")

import pytest
import numpy as np

from gnc_smoothie.sup_gauss_newton import SupGaussNewton
from gnc_smoothie.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie.welsch_influence_func import WelschInfluenceFunc

sys.path.append("../../pypi_package/tests")
from check_derivs import check_model_derivs

sys.path.append("../src/image_trs")
from trs import TRS

def randomM11() -> float:
    return 2.0*(np.random.rand() - 0.5)

def test_model_derivs():
    for test_idx in range(10):
        # ground-truth model
        model = np.array([randomM11(), randomM11(), randomM11(), randomM11()])

        # build data
        n_points = np.random.randint(5, 10)
        data = np.zeros((n_points,4))
        for i in range(n_points):
            for j in range(4):
                data[i][j] = randomM11()

        # check Python model
        model_instance = TRS()
        assert(check_model_derivs(model_instance, model, data))

def test_answer():
    np.random.seed(0) # We want the numbers to be the same on each run
    for test_idx in range(10):
        # ground-truth image transformation parameters
        model_gt = [2.0*(np.random.rand()-0.5), 2.0*(np.random.rand()-0.5), 2.0*(np.random.rand()-0.5), 2.0*(np.random.rand()-0.5)]

        # build good data
        n_good_point_pairs = np.random.randint(5, 20)
        n_outliers = int(np.random.rand()*n_good_point_pairs)

        data = np.zeros((n_good_point_pairs+n_outliers,4))
        for i in range(n_good_point_pairs):
            data[i][0] = 3*(np.random.rand() - 0.5)
            data[i][1] = 3*(np.random.rand() - 0.5)
            data[i][2] = model_gt[1]*data[i][0] - model_gt[0]*data[i][1] + model_gt[2]
            data[i][3] = model_gt[0]*data[i][0] + model_gt[1]*data[i][1] + model_gt[3]

        for i in range(n_outliers):
            data[n_good_point_pairs+i][0] = 3*(np.random.rand() - 0.5)
            data[n_good_point_pairs+i][1] = 3*(np.random.rand() - 0.5)
            data[n_good_point_pairs+i][2] = 3*(np.random.rand() - 0.5)
            data[n_good_point_pairs+i][3] = 3*(np.random.rand() - 0.5)

        param_instance = GNC_WelschParams(WelschInfluenceFunc(), 0.01, sigma_limit=50.0, num_sigma_steps=20)
        optimiser_instance = SupGaussNewton(param_instance, model_instance=TRS())
        assert(optimiser_instance.fit(data))
        model = optimiser_instance.final_model
        for i in range(4):
            assert(model[i] == pytest.approx(model_gt[i]))

if __name__ == "__main__":
    np.random.seed(43289) # We want the random numbers to be the same on each run
    test_model_derivs()
    test_answer()
