import sys
sys.path.append("../pypi_package/src")

import pytest
import numpy as np

from gnc_smoothie_philfm.sup_gauss_newton import SupGaussNewton
from gnc_smoothie_philfm.gnc_welsch_params import GNC_WelschParams
from gnc_smoothie_philfm.welsch_influence_func import WelschInfluenceFunc

sys.path.append("../examples/mean")
from gncs_robust_mean import RobustMean

def test_answer():
    np.random.seed(0) # We want the numbers to be the same on each run
    for test_idx in range(10):
        # ground-truth mean
        m_gt = 5.0*(np.random.rand()-0.5)

        # build good data
        n_good_points = np.random.randint(5, 20)
        n_outliers = int(np.random.rand()*n_good_points)

        data = np.zeros((n_good_points+n_outliers,1))
        for i in range(n_good_points):
            data[i][0] = m_gt

        for i in range(n_outliers):
            data[n_good_points+i][0] = 20*(np.random.rand() - 0.5)

        param_instance = GNC_WelschParams(WelschInfluenceFunc(), 0.01, 50.0, 20) # sigma_base, sigma_limit, num_sigma_steps
        optimiser_instance = SupGaussNewton(param_instance, RobustMean(), data)
        assert(optimiser_instance.run())
        model = optimiser_instance.final_model
        assert(model[0] == pytest.approx(m_gt))

if __name__ == "__main__":
    test_answer()
