import sys
import pytest
import numpy as np

sys.path.append("../pypi_package/src")
from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc

sys.path.append("../examples/line_fitting")
from line_fit import LineFit

def test_answer():
    np.random.seed(0) # We want the numbers to be the same on each run
    for test_idx in range(10):
        # ground-truth line parameters
        a_gt = 2.0*(np.random.rand()-0.5) # gradient
        b_gt = 10.0*(np.random.rand()-0.5) # intercept

        # build good data
        n_good_points = np.random.randint(5, 20)
        n_outliers = int(np.random.rand()*n_good_points)

        x_min = 10.0*np.random.rand()
        x_max = x_min + 3.0 + 5.0*np.random.rand()

        data = np.zeros((n_good_points+n_outliers,2))
        for i in range(n_good_points):
            data[i][0] = x_min + i*(x_max-x_min)/(n_good_points-1)
            data[i][1] = a_gt*data[i][0] + b_gt

        for i in range(n_outliers):
            data[n_good_points+i][0] = x_min + np.random.rand()*(x_max-x_min)
            data[n_good_points+i][1] = 20*(np.random.rand() - 0.5)

        param_instance = GNC_WelschParams(WelschInfluenceFunc(), 0.01, 50.0, 20) # sigma_base, sigma_limit, num_sigma_steps
        optimiser_instance = SupGaussNewton(param_instance, LineFit(), data)
        assert(optimiser_instance.run())
        model = optimiser_instance.final_model
        assert(model[0] == pytest.approx(a_gt))
        assert(model[1] == pytest.approx(b_gt))

if __name__ == "__main__":
    test_answer()
