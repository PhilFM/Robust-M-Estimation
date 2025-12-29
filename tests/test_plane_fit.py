import sys
import pytest
import numpy as np

sys.path.append("../pypi_package/src")
from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc

sys.path.append("../examples/plane_fitting")
from plane_fit import PlaneFit

def test_answer():
    np.random.seed(0) # We want the numbers to be the same on each run
    for test_idx in range(10):
        # ground-truth plane parameters
        a_gt = 2.0*(np.random.rand()-0.5) # x-gradient
        b_gt = 2.0*(np.random.rand()-0.5) # y-gradient
        c_gt = 10.0*(np.random.rand()-0.5) # intercept

        # build good data
        size_x = np.random.randint(5, 10)
        size_y = np.random.randint(5, 10)
        n_good_points = size_x*size_y
        n_outliers = int(np.random.rand()*n_good_points)

        x_min = 10.0*np.random.rand()
        x_max = x_min + 3.0 + 5.0*np.random.rand()
        y_min = 10.0*np.random.rand()
        y_max = y_min + 3.0 + 5.0*np.random.rand()

        data = np.zeros((n_good_points+n_outliers,3))
        for i in range(size_y):
            for j in range(size_x):
                idx = i*size_x+j
                data[idx][0] = x_min + j*(x_max-x_min)/(size_x-1)
                data[idx][1] = y_min + i*(y_max-y_min)/(size_y-1)
                data[idx][2] = a_gt*data[idx][0] + b_gt*data[idx][1] + c_gt

        for i in range(n_outliers):
            data[n_good_points+i][0] = x_min + np.random.rand()*(x_max-x_min)
            data[n_good_points+i][1] = y_min + np.random.rand()*(y_max-y_min)
            data[n_good_points+i][2] = 20*(np.random.rand() - 0.5)

        param_instance = GNC_WelschParams(WelschInfluenceFunc(), 0.01, 50.0, 20) # sigma_base, sigma_limit, num_sigma_steps
        optimiser_instance = SupGaussNewton(param_instance, PlaneFit(), data)
        assert(optimiser_instance.run())
        model = optimiser_instance.final_model
        assert(model[0] == pytest.approx(a_gt))
        assert(model[1] == pytest.approx(b_gt))
        assert(model[2] == pytest.approx(c_gt))

if __name__ == "__main__":
    test_answer()
